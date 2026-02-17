import sys
import csv
import torch
import pickle
import os
import json
import time
import secrets
import requests
from urllib.parse import urlencode
from datetime import datetime
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModel
from gensim.models import KeyedVectors
import numpy as np
import spotipy
from spotipy.exceptions import SpotifyException

csv.field_size_limit(sys.maxsize)
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

# Log file for saved playlists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
PLAYLIST_LOG_FILE = os.path.join(LOG_DIR, "saved_playlists.log")

# -------------------------------------------------------------------
# 1 · Paths to model, embeddings, and CSV files
# -------------------------------------------------------------------
ROOT = "/app"
MODEL_DIR        = ROOT + "/data/fine_tuned_model_no_scheduler_2"
EMBEDDINGS_FILE  = ROOT + "/data/playlists_embeddings_scheduler.pkl"

TRACKS_CSV       = ROOT + "/data/tracks.csv"
ITEMS_CSV        = ROOT + "/data/items.csv"
PLAYLISTS_CSV    = ROOT + "/data/playlists.csv"

# -------------------------------------------------------------------
# 2 · Global variables (loaded once)
# -------------------------------------------------------------------
_loaded = None
track_meta = {}
playlist_tracks = {}

SPOTIFY_SCOPES = "playlist-modify-public playlist-modify-private"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_ME_PLAYLISTS_URL = "https://api.spotify.com/v1/me/playlists"
SPOTIFY_PLAYLIST_ITEMS_URL_TEMPLATE = "https://api.spotify.com/v1/playlists/{playlist_id}/items"

# -------------------------------------------------------------------
# 3 · Static CSV loading (tracks + items)
# -------------------------------------------------------------------
print("Loading track metadata...")
with open(TRACKS_CSV, "r", encoding="utf8") as f:
    reader = csv.DictReader(f)
    for r in tqdm(reader, desc="tracks"):
        track_meta[r["track_uri"]] = {
            "track_name":  r["track_name"],
            "artist_name": r["artist_name"],
            "track_uri":   r["track_uri"]
        }

print("Loading playlist → track mapping...")
with open(ITEMS_CSV, "r", encoding="utf8") as f:
    reader = csv.DictReader(f)
    # i=0
    for r in tqdm(reader, desc="items"):
        # i+=1
        pid_str = r["pid"].strip()
        uri = r["track_uri"]
        if uri in track_meta:
            playlist_tracks.setdefault(pid_str, []).append(track_meta[uri])
        # if i == 1000000:
            # break

print("Static data loaded.\n")

# -------------------------------------------------------------------
# 4 · Load model, tokenizer, and embeddings (once)
# -------------------------------------------------------------------
def load():
    global _loaded
    if _loaded:
        return _loaded

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModel.from_pretrained(MODEL_DIR)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with open(EMBEDDINGS_FILE, "rb") as f:
        embdict = pickle.load(f)

    kv = KeyedVectors(vector_size=384)
    keys = list(embdict.keys())
    vectors = [embdict[k]["embedding"].astype(np.float32, copy=False) for k in keys]
    kv.add_vectors(keys, vectors)

    _loaded = (tokenizer, model, kv)
    return _loaded

# Force preloading at startup
print("Preloading tokenizer and model...")
load()
print("Model ready.\n")

# -------------------------------------------------------------------
# 5 · Encode playlist name → 384-D embedding
# -------------------------------------------------------------------
def embed_name(name: str, tokenizer, model) -> np.ndarray:
    with torch.no_grad():
        inputs = tokenizer(name, return_tensors="pt", truncation=True, padding=True).to(model.device)
        outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return emb

# -------------------------------------------------------------------
# 6 · Retrieve top-50 similar playlists
# -------------------------------------------------------------------
def find_similar(name: str, kv: KeyedVectors, tokenizer, model, topk: int = 50):
    emb = embed_name(name, tokenizer, model)
    return kv.similar_by_vector(emb, topn=topk)

# -------------------------------------------------------------------
# 7 · Aggregate top tracks by frequency
# -------------------------------------------------------------------
def top_tracks(similar_playlists, topk: int = 10):
    counter = Counter()
    for pid, _ in similar_playlists:
        pid_str = str(pid)
        for track in playlist_tracks.get(pid_str, []):
            uri = track["track_uri"]
            counter[uri] += 1
    return counter.most_common(topk)

# -------------------------------------------------------------------
# 7.5 · Log saved playlists
# -------------------------------------------------------------------
def log_saved_playlist(playlist_name, playlist_uri, playlist_id, track_uris, user_id=None):
    """Log saved playlist information to a file."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "playlist_name": playlist_name,
        "playlist_uri": playlist_uri,
        "playlist_id": playlist_id,
        "user_id": user_id,
        "track_uris": track_uris,
        "track_count": len(track_uris)
    }
    
    try:
        with open(PLAYLIST_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Error logging playlist: {e}")


def get_spotify_client_credentials():
    client_id = os.getenv("SPOTIFY_CLIENT_ID", "").strip()
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()
    return client_id, client_secret


def get_spotify_redirect_uri():
    configured = os.getenv("SPOTIFY_REDIRECT_URI", "").strip()
    if configured:
        return configured
    return request.host_url.rstrip("/") + url_for("spotify_callback")


def request_spotify_token(payload):
    client_id, client_secret = get_spotify_client_credentials()
    if not client_id or not client_secret:
        return None, "Spotify credentials are not configured on the server"

    payload = {
        **payload,
        "client_id": client_id,
        "client_secret": client_secret,
    }

    try:
        resp = requests.post(SPOTIFY_TOKEN_URL, data=payload, timeout=15)
    except Exception as exc:
        return None, f"Spotify token request failed: {exc}"

    if not resp.ok:
        return None, f"Spotify token request failed: {resp.text}"

    token_info = resp.json()
    token_info["expires_at"] = int(time.time()) + int(token_info.get("expires_in", 3600))
    return token_info, None


def get_valid_spotify_access_token():
    token_info = session.get("spotify_token_info")
    if not token_info:
        return None

    expires_at = int(token_info.get("expires_at", 0))
    if expires_at > int(time.time()) + 60:
        return token_info.get("access_token")

    refresh_token = token_info.get("refresh_token")
    if not refresh_token:
        session.pop("spotify_token_info", None)
        return None

    refreshed, err = request_spotify_token({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    })
    if err:
        session.pop("spotify_token_info", None)
        return None

    if "refresh_token" not in refreshed:
        refreshed["refresh_token"] = refresh_token
    if "scope" not in refreshed and "scope" in token_info:
        refreshed["scope"] = token_info["scope"]

    session["spotify_token_info"] = refreshed
    return refreshed.get("access_token")


def create_playlist_via_me_endpoint(access_token, playlist_name, is_public):
    payload = {
        "name": playlist_name,
        "public": is_public,
        "description": "Generated by LLM-Based Playlist Recommender",
    }
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    return requests.post(SPOTIFY_ME_PLAYLISTS_URL, headers=headers, json=payload, timeout=15)


def add_items_to_playlist_via_items_endpoint(access_token, playlist_id, track_uris):
    payload = {
        "uris": track_uris,
    }
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    url = SPOTIFY_PLAYLIST_ITEMS_URL_TEMPLATE.format(playlist_id=playlist_id)
    return requests.post(url, headers=headers, json=payload, timeout=15)

# -------------------------------------------------------------------
# 8 · Flask routes
# -------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend")
def recommend():
    name = request.args.get("playlist_name", "").strip()
    if not name:
        return jsonify({"error": "playlist_name required"}), 400

    tokenizer, model, kv = load()
    sim = find_similar(name, kv, tokenizer, model)
    topk_list = top_tracks(sim, topk=10)

    recommendations = []
    for uri, count in topk_list:
        meta = track_meta.get(uri, {})
        recommendations.append({
            "song":   meta.get("track_name", "Unknown"),
            "artist": meta.get("artist_name", "Unknown"),
            "uri":    uri,
            "count":  count
        })

    return jsonify({"recommendations": recommendations})

@app.route("/spotify-login")
def spotify_login():
    client_id, _ = get_spotify_client_credentials()
    if not client_id:
        return jsonify({"error": "SPOTIFY_CLIENT_ID is not configured on the server"}), 500

    redirect_uri = get_spotify_redirect_uri()
    state = secrets.token_urlsafe(16)
    session["spotify_oauth_state"] = state

    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": SPOTIFY_SCOPES,
        "state": state,
    }
    auth_url = f"{SPOTIFY_AUTH_URL}?{urlencode(params)}"
    return redirect(auth_url)


@app.route("/spotify-callback")
def spotify_callback():
    error = request.args.get("error")
    if error:
        return redirect(url_for("index"))

    expected_state = session.pop("spotify_oauth_state", None)
    received_state = request.args.get("state")
    if not expected_state or expected_state != received_state:
        return redirect(url_for("index"))

    code = request.args.get("code", "").strip()
    if not code:
        return redirect(url_for("index"))

    token_info, err = request_spotify_token({
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": get_spotify_redirect_uri(),
    })
    if err:
        return redirect(url_for("index"))

    session["spotify_token_info"] = token_info
    print("Spotify authentication successful.")
    print(f"Access token expires at {datetime.fromtimestamp(token_info['expires_at'])}")
    print(token_info)
    return redirect(url_for("index"))


@app.route("/spotify-auth-status")
def spotify_auth_status():
    access_token = get_valid_spotify_access_token()
    return jsonify({"connected": bool(access_token)})


@app.route("/spotify-logout", methods=["POST"])
def spotify_logout():
    session.pop("spotify_token_info", None)
    session.pop("spotify_oauth_state", None)
    return jsonify({"success": True})

@app.route("/save-playlist", methods=["POST"])
def save_playlist():
    try:
        data = request.get_json() or {}
        access_token = get_valid_spotify_access_token()
        playlist_name = data.get("playlist_name", "My Recommended Playlist")
        track_uris = data.get("track_uris", [])

        if not access_token:
            return jsonify({"error": "Spotify account is not connected"}), 401

        if not track_uris:
            return jsonify({"error": "Missing track_uris"}), 400

        token_info = session.get("spotify_token_info", {})
        scope_set = set((token_info.get("scope") or "").split())
        can_modify_public = "playlist-modify-public" in scope_set
        can_modify_private = "playlist-modify-private" in scope_set

        if not can_modify_public and not can_modify_private:
            return jsonify({
                "error": "Spotify token is missing playlist modification scopes. Please reconnect Spotify."
            }), 403

        # Create Spotify client with access token
        sp = spotipy.Spotify(auth=access_token)

        # Get current user
        user = sp.current_user()
        print(f"Saving playlist for Spotify user: {user['display_name']} ({user['id']})")
        user_id = user["id"]

        # Create playlist using POST /v1/me/playlists (prefer public, fallback to private on 403)
        playlist_public = can_modify_public
        response = create_playlist_via_me_endpoint(access_token, playlist_name, playlist_public)
        if response.status_code == 403 and playlist_public and can_modify_private:
            response = create_playlist_via_me_endpoint(access_token, playlist_name, False)

        if not response.ok:
            raise SpotifyException(
                response.status_code,
                -1,
                f"Failed to create playlist via /me/playlists: {response.text}"
            )

        playlist = response.json()

        playlist_id = playlist["id"]

        # Add tracks to playlist using POST /v1/playlists/{playlist_id}/items (in chunks of 100)
        for i in range(0, len(track_uris), 100):
            chunk = track_uris[i:i+100]
            add_response = add_items_to_playlist_via_items_endpoint(access_token, playlist_id, chunk)
            if not add_response.ok:
                raise SpotifyException(
                    add_response.status_code,
                    -1,
                    f"Failed to add items via /playlists/{{playlist_id}}/items: {add_response.text}"
                )

        playlist_url = playlist["external_urls"]["spotify"]
        playlist_uri = playlist["uri"]

        # Log the saved playlist
        log_saved_playlist(
            playlist_name=playlist_name,
            playlist_uri=playlist_uri,
            playlist_id=playlist_id,
            track_uris=track_uris,
            user_id=user_id
        )

        return jsonify({
            "success": True,
            "playlist_url": playlist_url,
            "playlist_id": playlist_id
        })

    except SpotifyException as e:
        return jsonify({"error": f"Spotify API error ({e.http_status}): {e.msg}"}), e.http_status or 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------------------------
# 9 · Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
