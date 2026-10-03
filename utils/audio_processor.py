import yt_dlp
from pydub import AudioSegment
import os

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR,exist_ok = True)

def download_youtube_audio(url :str) ->str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        # YouTube increasingly returns HTTP 403 for requests from cloud/
        # datacenter IP ranges (the kind most hosting platforms use) when
        # yt-dlp identifies itself with its default web-client signature.
        # Impersonating an Android client and retrying transient failures
        # is the standard, widely-used yt-dlp workaround for this — it asks
        # YouTube for a differently-signed stream that isn't gated the same
        # way, with no change to what gets downloaded.
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
        "http_headers": {"User-Agent": "com.google.android.youtube/19.09.37 (Linux; U; Android 14)"},
        "retries": 5,
        "fragment_retries": 5,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
        return filename
    except yt_dlp.utils.DownloadError as e:
        if "403" in str(e) or "Forbidden" in str(e):
            raise RuntimeError(
                "YouTube blocked this server's download request (HTTP 403). This is a "
                "temporary IP-level block on YouTube's side, not a problem with the link — "
                "wait a bit and try again, or upload the video/audio file directly instead."
            ) from e
        raise



def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz
    audio.export(output_path, format="wav")
    return output_path



def chunk_audio(wav_path : str , chunk_minutes : int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000 

    chunks = []

    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path , format = "wav")

        chunks.append(chunk_path)
    
    return chunks

def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks


