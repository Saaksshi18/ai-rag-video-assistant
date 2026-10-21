import yt_dlp
from pydub import AudioSegment
import os
import base64
import tempfile

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR,exist_ok = True)

# Error substrings that mean "this specific client identity / this specific
# request got challenged or blocked right now" — worth retrying with a
# different player client — as opposed to errors that mean the video/URL
# itself is the problem, which no amount of retrying will fix.
_RETRYABLE_MARKERS = (
    "403",
    "forbidden",
    "sign in to confirm",
    "not a bot",
    "confirm you're not a bot",
)


def _resolve_cookiefile() -> str | None:
    """
    Return a path to a cookies.txt file if one has been supplied, else None.

    YouTube increasingly challenges unauthenticated requests from cloud/
    datacenter IPs (Streamlit Community Cloud included) with "Sign in to
    confirm you're not a bot" — a check that spoofing the player client
    alone cannot pass. The documented, cost-free workaround is to attach
    cookies from a real, logged-in YouTube session so the request looks
    authenticated.

    Two ways to supply cookies, checked in order:
      1. YT_COOKIES_FILE — an absolute path to an existing cookies.txt
         (handy for local development).
      2. YT_COOKIES_B64 — the contents of a cookies.txt file, base64-encoded
         (the way to pass it as a Streamlit secret without committing the
         file to the repo). Decoded to a temp file on first use.
    """
    direct_path = os.environ.get("YT_COOKIES_FILE")
    if direct_path and os.path.isfile(direct_path):
        return direct_path

    b64_contents = os.environ.get("YT_COOKIES_B64")
    if b64_contents:
        try:
            decoded = base64.b64decode(b64_contents)
        except Exception:
            return None
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix="_yt_cookies.txt", mode="wb"
        )
        tmp.write(decoded)
        tmp.close()
        return tmp.name

    return None


def download_youtube_audio(url :str) ->str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    # YouTube's IP-based blocking of cloud/datacenter traffic targets some
    # yt-dlp "player client" identities more aggressively than others, and
    # which one is currently blocked shifts over time as YouTube adjusts
    # its detection. Betting on a single client (as the previous version
    # of this function did) means one block takes the whole feature down.
    # Trying several identities in sequence, falling through to the next
    # only on a block/challenge, is the resilient version of the same
    # workaround.
    client_attempts = ["android", "ios", "tv_embedded", "web", "mweb"]
    last_error = None
    cookiefile = _resolve_cookiefile()

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
        if cookiefile:
            ydl_opts["cookiefile"] = cookiefile
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                # FFmpegExtractAudio always outputs .wav, but the extension of the
                # downloaded source varies by client (.webm, .m4a, .mp4, ...).
                # Swap whatever it is for .wav instead of guessing a fixed list.
                filename = os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"
            return filename
        except yt_dlp.utils.DownloadError as e:
            last_error = e
            msg = str(e).lower()
            if any(marker in msg for marker in _RETRYABLE_MARKERS):
                # This client identity is blocked or challenged right now —
                # try the next one.
                continue
            # Any other failure (invalid URL, video unavailable, age/region
            # restricted, etc.) won't be fixed by switching client identity,
            # so fail immediately instead of burning time on four more tries.
            raise

    if cookiefile:
        raise RuntimeError(
            "YouTube blocked or challenged this server's download request across every "
            "client identity yt-dlp tried, even with cookies attached. The cookies may "
            "have expired — try exporting a fresh cookies.txt from a logged-in YouTube "
            "session — or upload the video/audio file directly instead."
        ) from last_error

    raise RuntimeError(
        "YouTube blocked or challenged this server's download request ('Sign in to "
        "confirm you're not a bot' / HTTP 403) across every client identity yt-dlp "
        "tried. This happens because the request looks unauthenticated to YouTube — "
        "the fix is to supply YouTube cookies from a logged-in session (set the "
        "YT_COOKIES_B64 secret) so the request looks like a real signed-in user. "
        "Until that's set up, upload the video/audio file directly instead."
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