FROM python:3.11-slim

WORKDIR /app

RUN pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

# Environment variables for Spotify API
# These should be passed at runtime with:
# docker run -e SPOTIFY_CLIENT_ID=... -e SPOTIFY_CLIENT_SECRET=... etc.
ENV SPOTIFY_CLIENT_ID=""
ENV SPOTIFY_CLIENT_SECRET=""
ENV SPOTIFY_REDIRECT_URI=""
ENV FLASK_SECRET_KEY=""

EXPOSE 8080
CMD ["python", "recommend_backend.py"]
