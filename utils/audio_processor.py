import yt_dlp
from pydub import AudioSegment
import os
import sys
import shutil
import base64
import tempfile

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# `pip install "yt-dlp[default,deno]"` drops a `deno` executable next to the
# Python interpreter (…/Scripts on Windows, …/bin elsewhere). That folder is
# only on PATH when the venv is "activated", which is not guaranteed when the
# app is launched by Streamlit Cloud or an IDE — so add it ourselves.
_py_bin_dir = os.path.dirname(sys.executable)
if _py_bin_dir and _py_bin_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _py_bin_dir + os.pathsep + os.environ.get("PATH", "")


def _js_runtimes() -> dict:
    """
    YouTube now serves a JavaScript challenge that yt-dlp must solve before it
    gets usable formats. Without a JS runtime you get "SABR-only" warnings,
    missing formats and "Sign in to confirm you're not a bot" even on a home IP.
    Tell yt-dlp about whichever runtime is installed (Deno preferred).
    """
    runtimes = {}
    for name in ("deno", "node", "bun"):
        path = shutil.which(name)
        if path:
            runtimes[name] = {"path": path}
    return runtimes


# Error substrings meaning "this client / this request was challenged or is
# missing formats right now" — worth retrying with a different player client —
# as opposed to errors about the video itself (private, removed, bad URL),
# which no amount of retrying will fix.
_RETRYABLE_MARKERS = (
    "403",
    "forbidden",
    "sign in to confirm",
    "not a bot",
    "po token",
    "sabr",
    "requested format is not available",
    "login_required",
    "no video formats",
    "unable to extract",
    "http error 429",
)


def _resolve_cookie_options() -> dict:
    """
    Authenticated requests are the most reliable way past YouTube's bot check.
    Supported, checked in this order:

      1. YT_COOKIES_FILE    - absolute path to a Netscape-format cookies.txt
      2. YT_COOKIES_B64     - base64 of a cookies.txt (use as a Streamlit
                              secret so the file never touches the repo)
      3. YT_COOKIES_BROWSER - browser name to read cookies from directly
                              (local use only), e.g. "firefox", "chrome",
                              "edge", "brave"

    Returns a dict of yt-dlp options (possibly empty).
    """
    direct_path = os.environ.get("YT_COOKIES_FILE")
    if direct_path and os.path.isfile(direct_path):
        return {"cookiefile": direct_path}

    b64_contents = os.environ.get("YT_COOKIES_B64")
    if b64_contents:
        try:
            # Tolerate output from `certutil -encode` (BEGIN/END header lines)
            # and any stray whitespace/newlines pasted into the secret.
            cleaned = "".join(
                line.strip()
                for line in b64_contents.splitlines()
                if not line.startswith("-----")
            )
            decoded = base64.b64decode(cleaned)
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix="_yt_cookies.txt", mode="wb"
            )
            tmp.write(decoded)
            tmp.close()
            return {"cookiefile": tmp.name}
        except Exception:
            pass

    browser = os.environ.get("YT_COOKIES_BROWSER")
    if browser:
        return {"cookiesfrombrowser": (browser.strip().lower(),)}

    return {}


def _is_retryable(msg: str) -> bool:
    msg = msg.lower()
    return any(marker in msg for marker in _RETRYABLE_MARKERS)


def download_youtube_audio(url: str) -> str:
    # Name files by video id: titles contain emoji / quotes / are very long,
    # which breaks paths on Windows and made us guess the final filename.
    output_path = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")

    cookie_opts = _resolve_cookie_options()
    runtimes = _js_runtimes()

    # First attempt lets yt-dlp pick its own current default clients (which is
    # what its maintainers keep working). Then fall back through specific
    # clients, because which ones YouTube blocks changes from week to week.
    client_attempts = [None, "tv", "mweb", "web_embedded", "android", "ios", "web"]
    last_error = None

    for client in client_attempts:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_path,
            "noplaylist": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
            "quiet": True,
            "no_warnings": True,
            "retries": 3,
            "fragment_retries": 3,
            # IPv6 ranges are flagged more aggressively than IPv4 ones.
            "source_address": "0.0.0.0",
        }
        ydl_opts.update(cookie_opts)
        if runtimes:
            ydl_opts["js_runtimes"] = runtimes
        if client:
            ydl_opts["extractor_args"] = {"youtube": {"player_client": [client]}}

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)

                # After FFmpegExtractAudio runs, yt-dlp records the final
                # (.wav) path here. Fall back to swapping the extension.
                final_path = None
                downloads = info.get("requested_downloads") or []
                if downloads:
                    final_path = downloads[0].get("filepath")
                if not final_path:
                    final_path = os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"
                elif not final_path.lower().endswith(".wav"):
                    final_path = os.path.splitext(final_path)[0] + ".wav"

            if os.path.isfile(final_path):
                return final_path
            last_error = RuntimeError(f"Download finished but {final_path} was not found")
            continue
        except yt_dlp.utils.DownloadError as e:
            last_error = e
            if _is_retryable(str(e)):
                continue
            raise  # private / removed / bad URL — retrying won't help

    # Every attempt failed. Say what is actually missing.
    hints = []
    if not runtimes:
        hints.append(
            "No JavaScript runtime was found. Run: pip install -U \"yt-dlp[default,deno]\" "
            "(or install Deno/Node.js) — current YouTube downloads need one."
        )
    if not cookie_opts:
        hints.append(
            "No YouTube cookies are configured. Set YT_COOKIES_BROWSER (local, e.g. firefox) "
            "or YT_COOKIES_B64 / YT_COOKIES_FILE."
        )
    hints.append("Make sure yt-dlp is current: pip install -U yt-dlp")
    hints.append("Cloud server IPs are often blocked by YouTube — uploading the file always works.")

    raise RuntimeError(
        "Could not download this video from YouTube after trying every client. "
        + " ".join(hints)
        + f" (last error: {last_error})"
    ) from last_error


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # 16khz
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")

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