let spotifyConnected = false;
let currentPlaylistName = '';
let currentRecommendations = [];

function getSpotifyIconUrl() {
    const results = document.getElementById('results');
    if (!results) {
        return '';
    }
    return results.dataset.spotifyIcon || '';
}

function updateSpotifyUI() {
    const section = document.getElementById('spotify-section');
    const authBtn = document.getElementById('auth-button');
    const logoutBtn = document.getElementById('logout-button');
    const saveBtn = document.getElementById('save-button');
    const status = document.getElementById('spotify-status');

    if (!section || !authBtn || !logoutBtn || !saveBtn || !status) {
        return;
    }

    section.classList.add('active');

    if (spotifyConnected) {
        authBtn.style.display = 'none';
        logoutBtn.style.display = 'block';
        saveBtn.style.display = currentRecommendations.length > 0 ? 'block' : 'none';
        status.textContent = '✓ Connected to Spotify';
        return;
    }

    authBtn.style.display = 'block';
    logoutBtn.style.display = 'none';
    saveBtn.style.display = 'none';
    status.textContent = 'Connect your Spotify account to save playlists';
}

function authorizeSpotify() {
    window.location.href = '/spotify-login';
}

async function fetchSpotifyAuthStatus() {
    try {
        const res = await fetch('/spotify-auth-status');
        const data = await res.json();
        spotifyConnected = Boolean(res.ok && data.connected);
    } catch {
        spotifyConnected = false;
    }

    updateSpotifyUI();
}

async function disconnectSpotify() {
    try {
        const res = await fetch('/spotify-logout', { method: 'POST' });
        if (!res.ok) {
            alert('Error disconnecting Spotify.');
            return;
        }
        spotifyConnected = false;
        updateSpotifyUI();
    } catch {
        alert('Error disconnecting Spotify.');
    }
}

async function savePlaylistToSpotify() {
    if (!spotifyConnected || !currentPlaylistName || currentRecommendations.length === 0) {
        alert('Missing playlist data');
        return;
    }

    const saveBtn = document.getElementById('save-button');
    if (!saveBtn) {
        return;
    }

    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving...';

    try {
        const res = await fetch('/save-playlist', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                playlist_name: currentPlaylistName,
                track_uris: currentRecommendations.map(r => r.uri)
            })
        });

        const data = await res.json();
        if (res.ok && data.playlist_url) {
            window.open(data.playlist_url, '_blank');
            alert('Playlist saved successfully!');
        } else {
            alert('Error saving playlist: ' + (data.error || 'Unknown error'));
        }
    } catch (err) {
        alert('Error saving playlist: ' + err.message);
    } finally {
        saveBtn.disabled = false;
        saveBtn.textContent = 'Save Playlist to Spotify';
    }
}

async function ask() {
    const input = document.getElementById('q');
    const recTitle = document.getElementById('rec-title');
    const results = document.getElementById('results');

    if (!input || !recTitle || !results) {
        return;
    }

    const q = input.value.trim();
    if (!q) {
        return;
    }

    currentPlaylistName = q;
    recTitle.textContent = 'Loading...';
    results.innerHTML = '';

    try {
        const res = await fetch(`/recommend?playlist_name=${encodeURIComponent(q)}`);
        const data = await res.json();

        if (!res.ok || data.error) {
            recTitle.textContent = 'Error fetching recommendations.';
            return;
        }

        currentRecommendations = data.recommendations;
        recTitle.textContent = `Top 10 recommendations for "${q}". Click the green Spotify icon to play a track.`;
        updateSpotifyUI();
        render(data.recommendations);
    } catch {
        recTitle.textContent = 'Unexpected error.';
    }
}

function render(recs) {
    const ul = document.getElementById('results');
    if (!ul) {
        return;
    }

    const iconUrl = getSpotifyIconUrl();
    ul.innerHTML = '';

    recs.forEach((r, i) => {
        const li = document.createElement('li');
        li.innerHTML = `<div class="song-info"><span class="song-title">${i + 1}. ${r.song}</span><span class="song-artist">— ${r.artist}</span></div>`;

        const btn = document.createElement('button');
        btn.className = 'play-button';
        btn.innerHTML = `<img src="${iconUrl}" alt="Play on Spotify">`;
        btn.onclick = () => {
            if (r.uri) {
                window.open(`https://open.spotify.com/track/${r.uri.split(':').pop()}`, '_blank');
            }
        };

        li.appendChild(btn);
        ul.appendChild(li);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('q');
    const authBtn = document.getElementById('auth-button');
    const logoutBtn = document.getElementById('logout-button');
    const saveBtn = document.getElementById('save-button');

    if (input) {
        input.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                ask();
            }
        });
    }

    if (authBtn) {
        authBtn.addEventListener('click', authorizeSpotify);
    }

    if (logoutBtn) {
        logoutBtn.addEventListener('click', disconnectSpotify);
    }

    if (saveBtn) {
        saveBtn.addEventListener('click', savePlaylistToSpotify);
    }

    fetchSpotifyAuthStatus();
});
