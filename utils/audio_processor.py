import yt_dlp
from pydub import AudioSegment
import os

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR,exist_ok = True)

def download_youtube_audio(url :str) ->str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    # YouTube's IP-based blocking of cloud/datacenter traffic targets some
    # yt-dlp "player client" identities more aggressively than others, and
    # which one is currently blocked shifts over time as YouTube adjusts
    # its detection. Betting on a single client (as the previous version
    # of this function did) means one block takes the whole feature down.
    # Trying several identities in sequence, falling through to the next
    # only on a 403, is the resilient version of the same workaround.
    client_attempts = ["android", "ios", "tv_embedded", "web", "mweb"]
    last_error = None

    for client in client_attempts:
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
            "extractor_args": {"youtube": {"player_client": [client]}},
            "retries": 2,
            "fragment_retries": 2,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
            return filename
        except yt_dlp.utils.DownloadError as e:
            last_error = e
            msg = str(e)
            if "403" in msg or "Forbidden" in msg:
                # This client identity is blocked right now — try the next one.
                continue
            # Any other failure (invalid URL, video unavailable, age/region
            # restricted, etc.) won't be fixed by switching client identity,
            # so fail immediately instead of burning time on four more tries.
            raise

    raise RuntimeError(
        "YouTube blocked this server's download request across every client identity "
        "yt-dlp tried (HTTP 403). This is an IP-level block on YouTube's side that is "
        "currently affecting this server, not a problem with the link — wait a few "
        "minutes and try again, or upload the video/audio file directly instead."
    ) from last_error



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


