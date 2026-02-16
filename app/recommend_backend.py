import sys
import csv
import torch
import pickle
import os
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModel
from gensim.models import KeyedVectors
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth

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
MODEL_DIR        = "/app/data/fine_tuned_model_no_scheduler_2"
EMBEDDINGS_FILE  = "/app/data/playlists_embeddings_scheduler.pkl"

TRACKS_CSV       = "/app/data/tracks.csv"
ITEMS_CSV        = "/app/data/items.csv"
PLAYLISTS_CSV    = "/app/data/playlists.csv"

# -------------------------------------------------------------------
# 2 · Global variables (loaded once)
# -------------------------------------------------------------------
_loaded = None
track_meta = {}
playlist_tracks = {}

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
    for r in tqdm(reader, desc="items"):
        pid_str = r["pid"].strip()
        uri = r["track_uri"]
        if uri in track_meta:
            playlist_tracks.setdefault(pid_str, []).append(track_meta[uri])

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

@app.route("/save-playlist", methods=["POST"])
def save_playlist():
    try:
        data = request.get_json()
        access_token = data.get("access_token")
        playlist_name = data.get("playlist_name", "My Recommended Playlist")
        track_uris = data.get("track_uris", [])

        if not access_token or not track_uris:
            return jsonify({"error": "Missing access_token or track_uris"}), 400

        # Create Spotify client with access token
        sp = spotipy.Spotify(auth=access_token)

        # Get current user
        user = sp.current_user()
        user_id = user["id"]

        # Create playlist
        playlist = sp.user_playlist_create(
            user_id,
            playlist_name,
            public=True,
            description=f"Generated by LLM-Based Playlist Recommender"
        )

        playlist_id = playlist["id"]

        # Add tracks to playlist (in chunks of 100)
        for i in range(0, len(track_uris), 100):
            chunk = track_uris[i:i+100]
            sp.playlist_add_items(playlist_id, chunk)

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

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------------------------
# 9 · Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
