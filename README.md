# Playlist Recommendation Demo

This project is a demo application that recommends songs based on a user-provided playlist title.
It uses a fine-tuned transformer model trained on the Million Playlist Dataset to generate real-time suggestions that match the theme of the input.

The system is fully packaged in a Docker image and includes:

- A Flask backend API
- A clean, interactive web interface
- Preloaded data, embeddings, and model (no extra setup required)

**Related links**

- [Main project repository](https://github.com/elea-vellard/LLM-Playlist-Recommender) with the main method
- [Online demo](https://playlist-recommendation.tools.eurecom.fr/)

---

## Setup: Spotify API Credentials

This application uses the **Spotify Web API** to create and populate playlists. You need to register your application on Spotify Developer to get credentials.

### Get your Spotify credentials:

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Log in or create a free Spotify account
3. Create a new application
4. Accept the terms and create the app
5. You'll receive:
   - **Client ID**
   - **Client Secret**
6. In the app settings, add a **Redirect URI** (e.g., `http://localhost:5000/` for local development or your production URL)

### Set environment variables:

Create a `.env` file or set these variables before running:

```bash
export SPOTIFY_CLIENT_ID="your_client_id_here"
export SPOTIFY_CLIENT_SECRET="your_client_secret_here"
export SPOTIFY_REDIRECT_URI="http://localhost:5000/"
export FLASK_SECRET_KEY="your_secret_key_here"
```

For production, use your actual domain in the redirect URI.

---

## Run the demo locally

### 1. Download the data

First, populate the `app/data/` directory:

- apply `transform-dataset/json2csv.py` on the [Million Playlist Dataset](https://www.kaggle.com/datasets/himanshuwagh/spotify-million) and put the three obtained csv in the directory;
- download the model in the [Zenodo repository](https://zenodo.org/records/15837980) and decompress in the directory.

Make sure your project follows this structure:

```bash
project-root/
├── app/
│   ├── recommend_backend.py                  # Back-end logic (Flask API)
│   ├── templates/
│   │   └── index.html                        # Front-end (UI)
│   ├── static/
│   │   └── images/
│   │       ├── github-logo.png
│   │       └── spotify.png
│   └── data/
│       ├── tracks.csv
│       ├── items.csv
│       ├── playlists.csv
│       ├── playlists_embeddings_scheduler.pkl
│       └── fine_tuned_model_no_scheduler_2/  # Fine-tuned model
├── .env                                      # Environment variables (optional)
├── Dockerfile
├── requirements.txt
└── README.md
```

### Running locally (without Docker)

```bash
# Set environment variables
source env.txt  # or export them manually

# Install dependencies
pip install -r requirements.txt

# Run the Flask development server
cd app/
python recommend_backend.py
```

The app will be available at `http://localhost:8080`

## Features

### Playlist Logging

All saved playlists are automatically logged to `saved_playlists.log` in JSON format. Each log entry contains:
- **timestamp**: When the playlist was created (UTC)
- **playlist_name**: Title of the playlist
- **playlist_uri**: Spotify URI (e.g., `spotify:playlist:xxx`)
- **playlist_id**: Spotify playlist ID
- **user_id**: Spotify user ID who created it
- **track_uris**: Array of all track URIs in the playlist
- **track_count**: Number of tracks

Example log entry:
```json
{"timestamp": "2026-02-16T10:30:45.123456", "playlist_name": "Chill evening vibes", "playlist_uri": "spotify:playlist:abc123", "playlist_id": "abc123", "user_id": "user123", "track_uris": ["spotify:track:xxx", "spotify:track:yyy"], "track_count": 10}
```

### 2. Build the image

From the root of the project, run:

```bash
docker build -t playlist-recommendation:latest .
```

### 3. Run the container

Pass your Spotify credentials as environment variables:

```bash
docker run --rm \
  -p 8080:8080 \
  -v $(pwd)/logs:/app/logs \
  -e SPOTIFY_CLIENT_ID="your_client_id" \
  -e SPOTIFY_CLIENT_SECRET="your_client_secret" \
  -e SPOTIFY_REDIRECT_URI="https://playlist-recommendation.tools.eurecom.fr" \
  -e FLASK_SECRET_KEY="your_secret_key" \
  playlist-recommendation:latest
```

Or load from `.env` file:

```bash
docker run --name playlist-recommendation --restart=unless-stopped -p 9981:8080 -v $(pwd)/logs:/app/logs -v $(pwd)/app/data:/app/data --env-file .env playlist-recommendation:latest
```

**Note:** The `-v $(pwd)/logs:/app/logs` mount ensures the `saved_playlists.log` file persists on your host machine.

---

## Citation

If you use this software, please cite ([bib file](https://raw.githubusercontent.com/elea-vellard/LM-Playlist-Recommender/refs/heads/main/vellardcharolois2025demo.bib)):

Eléa Vellard, Enzo Charolois–Pasqua, Youssra Rebboud, Pasquale Lisena,
and Raphael Troncy. 2025. Interactive Playlist Generation from Titles. In
Proceedings of the Nineteenth ACM Conference on Recommender Systems
(RecSys ’25), September 22–26, 2025, Prague, Czech Republic. ACM, New York,
NY, USA, 3 pages. https://doi.org/10.1145/3705328.3759336