from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import requests
from bs4 import BeautifulSoup
import io
import re
import urllib.parse
import tempfile
import os

app = FastAPI(title="GELBET Sonic Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def fetch_audio_preview(query: str):
    url = f"https://itunes.apple.com/search?term={urllib.parse.quote(query)}&entity=song&limit=1"
    try:
        response = requests.get(url, timeout=5).json()
        if response['resultCount'] > 0:
            track = response['results'][0]
            return {
                "audio_url": track.get("previewUrl"),
                "artist": track.get("artistName"),
                "song": track.get("trackName"),
                "genre": track.get("primaryGenreName")
            }
    except Exception as e:
        print(f"[SONIC ERROR] Fetch audio: {e}")
    return None

def fetch_lyrics(artist: str, song: str):
    clean_artist = re.sub(r'[^a-zA-Z0-9]', '', artist.lower())
    clean_song = re.sub(r'[^a-zA-Z0-9]', '', song.lower())
    url = f"https://www.azlyrics.com/lyrics/{clean_artist}/{clean_song}.html"
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            divs = soup.find_all('div', class_=False, id=False)
            for div in divs:
                if len(div.text) > 100 and "Submit Corrections" not in div.text:
                    return div.text.strip()
    except Exception:
        pass
    return "instrumental void silence atmospheric"

def analyze_hypertext(lyrics: str):
    text = lyrics.lower()
    words = re.findall(r'\b\w+\b', text)
    if not words:
        return {"entropy": 0.5, "vowel_darkness": 0.5, "syllabic_density": 0.1}

    unique_words = set(words)
    entropy = np.clip(len(unique_words) / (len(words) + 1e-5) * 2.0, 0.0, 1.0) 
    
    heavy_vowels = len(re.findall(r'[ouоу]', text))
    light_vowels = len(re.findall(r'[ieие]', text))
    total_vowels = heavy_vowels + light_vowels + 1e-5
    vowel_darkness = np.clip(heavy_vowels / total_vowels, 0.0, 1.0)

    lines = [line for line in lyrics.split('\n') if len(line.strip()) > 0]
    avg_words_per_line = len(words) / (len(lines) + 1e-5)
    syllabic_density = np.clip(avg_words_per_line / 12.0, 0.0, 1.0)

    return {"entropy": entropy, "vowel_darkness": vowel_darkness, "syllabic_density": syllabic_density}

# 🔥 ИСПРАВЛЕНА РАБОТА С ПАМЯТЬЮ (ФИЗИЧЕСКИЙ ФАЙЛ ДЛЯ FFMPEG) 🔥
def analyze_acoustic_physics(audio_bytes: bytes):
    # Создаем временный файл на диске сервера
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # Librosa (и ffmpeg под капотом) спокойно читает физический файл
        y, sr = librosa.load(tmp_path, sr=22050, duration=30.0)
        
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        brightness = np.clip(np.mean(centroid) / 3000.0, 0.0, 1.0)
        
        rms = librosa.feature.rms(y=y)[0]
        dynamics = np.clip(np.std(rms) / (np.mean(rms) + 1e-5), 0.0, 1.0)
        
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        gilmour_peak = np.clip((np.max(onset_env) - np.mean(onset_env)) / (np.std(onset_env) + 1e-5) / 10.0, 0.0, 1.0)
        
        zero_crossings = librosa.feature.zero_crossing_rate(y)[0]
        numbness = np.clip(1.0 - np.mean(zero_crossings) * 5.0, 0.0, 1.0)

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        pink_noise = np.clip(np.mean(rolloff) / 8000.0, 0.0, 1.0)

        return {"brightness": brightness, "dynamics": dynamics, "gilmour_peak": gilmour_peak, "numbness": numbness, "pink_noise": pink_noise}
    
    except Exception as e:
        print(f"[ACOUSTIC ERROR] {e}")
        return {"brightness": 0.5, "dynamics": 0.5, "gilmour_peak": 0.5, "numbness": 0.5, "pink_noise": 0.5}
    
    finally:
        # ОБЯЗАТЕЛЬНО удаляем временный файл, чтобы не устроить утечку дисковой памяти
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def translate_to_visual_tensor(acoustic: dict, text_math: dict, genre: str):
    tensor = np.zeros(32)
    tensor[0] = (acoustic["brightness"] * 0.7) + ((1.0 - text_math["vowel_darkness"]) * 0.3)
    tensor[1] = acoustic["dynamics"]
    tensor[2] = text_math["entropy"]
    tensor[3] = text_math["syllabic_density"]
    tensor[5] = 1.0 - acoustic["brightness"]
    tensor[8] = acoustic["numbness"]
    
    if "rock" in genre.lower() or "metal" in genre.lower():
        tensor[11] = 0.8 
    elif "electronic" in genre.lower() or "pop" in genre.lower():
        tensor[11] = 0.2 
        
    tensor[15] = acoustic["pink_noise"]
    tensor[24] = text_math["entropy"]
    tensor[26] = acoustic["numbness"]
    tensor[27] = acoustic["gilmour_peak"]

    for i in range(32):
        if tensor[i] == 0:
            tensor[i] = np.random.uniform(0.3, 0.7)

    return [round(float(x), 4) for x in tensor]

@app.get("/")
def health():
    return {"status": "SONIC_ENGINE_ONLINE"}

@app.post("/api/resonate")
async def resonate_audio(request: Request):
    try:
        payload = await request.json()
        query = payload.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Missing query")

        logs = [f"[SONIC] Инициация поиска сигнала: {query}"]

        track_info = fetch_audio_preview(query)
        if not track_info:
            return {"status": "error", "message": "Signal not found in open networks."}
        
        logs.append(f"[SONIC] Захвачен сигнал: {track_info['artist']} - {track_info['song']} ({track_info['genre']})")

        logs.append("[SONIC] Сканирование гипертекста...")
        lyrics = fetch_lyrics(track_info['artist'], track_info['song'])
        text_math = analyze_hypertext(lyrics)
        logs.append(f"[SONIC] Энтропия текста: {text_math['entropy']:.2f} | Тяжесть гласных: {text_math['vowel_darkness']:.2f}")

        logs.append("[SONIC] Захват аудио-буфера. Извлечение спектра...")
        audio_req = requests.get(track_info['audio_url'])
        acoustic_math = analyze_acoustic_physics(audio_req.content)
        logs.append(f"[SONIC] Gilmour Peak: {acoustic_math['gilmour_peak']:.2f} | Numbness: {acoustic_math['numbness']:.2f}")

        logs.append("[SONIC] Вычисление синестезии. Генерация 32D тензора...")
        visual_tensor = translate_to_visual_tensor(acoustic_math, text_math, track_info['genre'])

        anchor = f"{track_info['artist']} {track_info['song']}"

        return {
            "status": "success",
            "anchor": anchor,
            "visual_tensor": visual_tensor,
            "logs": logs
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
