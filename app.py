"""
AI Video Assistant — Streamlit frontend.

FRONTEND-ONLY REDESIGN. Every backend call below is identical to the
previous version of this file: process_input, transcribe_all,
generate_title, summarize, extract_action_items / extract_key_decisions /
extract_questions, build_rag_chain, ask_question. Nothing in core/ or
utils/ was changed — same imports, same call signatures, same pipeline
order, same environment variables (GROQ_API_KEY, SARVAM_API_KEY,
WHISPER_MODEL, SARVAM_STT_MODEL).
"""
import html
import json
import os
import re
import tempfile
import threading
import time
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from fpdf import FPDF
from pydub import AudioSegment
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="▸",
    layout="wide",
    initial_sidebar_state="expanded",
)

GITHUB_URL = "https://github.com/Saaksshi18/ai-rag-video-assistant"
SUPPORTED_UPLOAD_TYPES = ["mp4", "mp3", "wav", "m4a"]
SUGGESTED_QUESTIONS = [
    "What decisions were made?",
    "What are my action items?",
    "When is the project deadline?",
    "What concerns were raised?",
    "Summarise the deployment discussion.",
]
WAITING_MESSAGES = [
    "Something's cooking — good results take a moment \U0001f373",
    "Please wait, patience gives fruitful results \U0001f331",
    "Listening closely so nothing gets missed \U0001f3a7",
    "Reading between the lines of your transcript \U0001f4dd",
    "No, it hasn't frozen — it's genuinely thinking \U0001f916",
    "Good things come to those who let the model work ✨",
    "Almost there — stitching the pieces together \U0001f9f5",
]
PIPELINE_STEPS = [
    ("audio", "Audio processing"),
    ("transcript", "Transcription"),
    ("title", "Title generation"),
    ("summary", "Summarisation"),
    ("extract", "Extraction"),
    ("rag", "RAG engine"),
]

# ─────────────────────────────────────────────────────────────────────────
# Design system — restrained dark theme, Inter type, subtle borders only
# ─────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --bg: #0c0c0f;
        --surface: #131318;
        --surface-2: #191a20;
        --surface-3: #202127;
        --border: #26272e;
        --border-soft: #1d1e24;
        --accent: #5fe3a0;
        --accent-soft: rgba(95, 227, 160, 0.13);
        --accent-2: #4fb8c9;
        --accent-2-soft: rgba(79, 184, 201, 0.14);
        --text: #eeeeec;
        --text-muted: #96969f;
        --text-faint: #5c5c66;
        --success: #5fe3a0;
        --warning: #e0b85c;
        --radius: 10px;
    }

    html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; color: var(--text); }
    .stApp { background: var(--bg); }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-faint); }

    section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border-soft); }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.25rem; }

    h1, h2, h3, h4 { font-family: 'Inter', sans-serif; letter-spacing: -0.01em; }

    .stTextInput > div > div > input,
    .stTextArea textarea,
    div[data-baseweb="select"] > div {
        background: var(--surface-2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-soft) !important;
    }

    .stButton > button {
        background: var(--surface-2);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.875rem;
        transition: border-color 0.15s ease, background 0.15s ease;
    }
    .stButton > button:hover { border-color: var(--accent); background: var(--surface-3); }
    .stButton > button:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
    button[kind="primary"] { background: var(--accent) !important; border: 1px solid var(--accent) !important; color: #06110b !important; font-weight: 600 !important; }
    button[kind="primary"]:hover { background: #7ce9b3 !important; border-color: #7ce9b3 !important; }

    div[role="radiogroup"] {
        display: inline-flex; gap: 0.2rem;
        background: var(--surface-2); border: 1px solid var(--border);
        border-radius: 8px; padding: 3px;
    }
    div[role="radiogroup"] label { margin: 0 !important; padding: 0.35rem 0.85rem !important; border-radius: 6px !important; font-size: 0.82rem !important; }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border-soft); }
    .stTabs [data-baseweb="tab"] { color: var(--text-muted); font-size: 0.85rem; }
    .stTabs [aria-selected="true"] { color: var(--text) !important; border-bottom: 2px solid var(--accent) !important; }

    [data-testid="stVerticalBlockBorderWrapper"] > div { border-color: var(--border-soft) !important; border-radius: var(--radius) !important; }

    hr { border-top: 1px solid var(--border-soft) !important; margin: 1.25rem 0 !important; }

    .eyebrow { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.09em; text-transform: uppercase; color: var(--accent-2); margin-bottom: 0.4rem; }
    .hero-h1 { font-size: clamp(2.4rem, 5.6vw, 4rem); font-weight: 800; line-height: 1.12; color: var(--text); margin: 0 0 0.6rem 0; letter-spacing: -0.02em; }
    .hero-sub { color: var(--text-muted); font-size: 1.05rem; max-width: 36rem; line-height: 1.6; }

    .surface-panel { background: var(--surface); border: 1px solid var(--border-soft); border-radius: var(--radius); padding: 1.1rem 1.3rem; }

    .section-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-faint); margin: 0.2rem 0 0.6rem 0; }

    .meta-row { display: flex; flex-wrap: wrap; gap: 1.6rem; margin-top: 0.5rem; }
    .meta-chip { display: flex; flex-direction: column; gap: 0.1rem; }
    .meta-chip .meta-label { font-size: 0.66rem; color: var(--text-faint); text-transform: uppercase; letter-spacing: .06em; }
    .meta-chip .meta-value { font-size: 0.86rem; color: var(--text); font-weight: 500; }

    .divider-item { border-bottom: 1px solid var(--border-soft); padding: 0.8rem 0; }
    .divider-item:last-child { border-bottom: none; }
    .divider-item-body { font-size: 0.85rem; color: var(--text-muted); line-height: 1.6; white-space: pre-wrap; }

    .sidebar-nav-title { font-size: 0.95rem; font-weight: 600; color: var(--text); margin-bottom: 0.1rem; }
    .sidebar-nav-sub { font-size: 0.72rem; color: var(--text-faint); margin-bottom: 1.1rem; }
    .sidebar-section-label { font-size: 0.66rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-faint); margin: 1.1rem 0 0.4rem 0; }
    .history-empty { font-size: 0.8rem; color: var(--text-faint); padding: 0.2rem 0.1rem; }

    mark { background: var(--accent-soft); color: var(--text); padding: 0 2px; border-radius: 3px; }

    /* ── Top header bar ── */
    .topbar {
        display: flex; align-items: center; justify-content: space-between;
        padding: 0.9rem 0 1.1rem 0; border-bottom: 1px solid var(--border-soft);
        margin-bottom: 1.6rem;
    }
    .topbar-brand { display: flex; align-items: center; gap: 0.6rem; }
    .topbar-logo {
        width: 30px; height: 30px; border-radius: 8px;
        background: var(--accent); color: #06110b;
        display: flex; align-items: center; justify-content: center;
        font-weight: 800; font-size: 0.95rem;
    }
    .topbar-title { font-size: 1.02rem; font-weight: 700; color: var(--text); }
    .topbar-link {
        display: inline-flex; align-items: center; gap: 0.4rem;
        border: 1px solid var(--border); border-radius: 8px;
        padding: 0.4rem 0.85rem; font-size: 0.82rem; font-weight: 500;
        color: var(--text); text-decoration: none !important;
        transition: border-color 0.15s ease;
    }
    .topbar-link:hover { border-color: var(--accent); color: var(--accent); }

    /* ── Pipeline checklist ── */
    .pl-row { display: flex; align-items: center; gap: 0.65rem; padding: 0.5rem 0; font-size: 0.88rem; }
    .pl-icon { width: 18px; height: 18px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; }
    .pl-done .pl-icon { color: var(--success); }
    .pl-active .pl-icon { color: var(--accent); animation: spin 0.9s linear infinite; }
    .pl-pending .pl-icon { color: var(--text-faint); }
    .pl-done .pl-label { color: var(--text-muted); }
    .pl-active .pl-label { color: var(--text); font-weight: 600; }
    .pl-pending .pl-label { color: var(--text-faint); }
    @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

    .wait-line {
        font-size: 0.82rem; color: var(--text-muted);
        margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px dashed var(--border-soft);
    }
    .wait-timer { font-family: 'IBM Plex Mono', monospace; color: var(--accent); font-weight: 600; }

    /* ── Skeleton shimmer for "still generating" placeholders ── */
    .skeleton-bar {
        height: 10px; border-radius: 4px; margin: 8px 0;
        background: linear-gradient(90deg, var(--surface-2) 25%, var(--surface-3) 37%, var(--surface-2) 63%);
        background-size: 400% 100%;
        animation: shimmer 1.6s ease infinite;
    }
    @keyframes shimmer { 0% { background-position: 100% 50%; } 100% { background-position: 0 50%; } }

    /* ── Fade-in ── */
    .fade-in { animation: fadeIn 0.4s ease; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

    /* ── Output table (action items) ── */
    .out-table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
    .out-table th {
        text-align: left; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em;
        color: var(--text-faint); font-weight: 600; padding: 0 0 0.5rem 0; border-bottom: 1px solid var(--border-soft);
    }
    .out-table td { padding: 0.55rem 0.6rem 0.55rem 0; border-bottom: 1px solid var(--border-soft); color: var(--text-muted); vertical-align: top; }
    .out-table tr:last-child td { border-bottom: none; }
    .out-table td.out-task { color: var(--text); }
    .out-table td.out-due { font-family: 'IBM Plex Mono', monospace; color: var(--accent-2); white-space: nowrap; }

    .bullet-list { list-style: none; margin: 0; padding: 0; }
    .bullet-list li { display: flex; gap: 0.55rem; padding: 0.5rem 0; font-size: 0.85rem; color: var(--text-muted); border-bottom: 1px solid var(--border-soft); }
    .bullet-list li:last-child { border-bottom: none; }
    .bullet-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); margin-top: 0.4rem; flex-shrink: 0; }
    .bullet-dot.amber { background: var(--warning); }

    .meta-line { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: var(--text-faint); margin-top: 0.3rem; }

    @media (max-width: 640px) {
        .hero-h1 { font-size: 2rem; }
        .surface-panel { padding: 0.9rem; }
        .topbar { flex-wrap: wrap; gap: 0.6rem; }
    }
    </style>
    """, unsafe_allow_html=True)


def esc(text) -> str:
    return html.escape(text or "")


def format_duration(total_seconds) -> str:
    if not total_seconds or total_seconds <= 0:
        return "—"
    total_seconds = int(total_seconds)
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


class LiveTimer:
    """Ticks a placeholder with elapsed time + a rotating reassuring message
    while the pipeline runs, so a multi-minute analysis never looks stalled.
    Runs in a background thread attached to Streamlit's script context;
    always stopped in a `finally` block by the caller."""

    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.start = time.time()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        ctx = get_script_run_ctx()
        if ctx is not None:
            add_script_run_ctx(self._thread, ctx)

    def start_timer(self):
        self._thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            elapsed = int(time.time() - self.start)
            msg = WAITING_MESSAGES[(elapsed // 4) % len(WAITING_MESSAGES)]
            try:
                self.placeholder.markdown(
                    f'<div class="wait-line">⏱️ <span class="wait-timer">{format_duration(elapsed) if elapsed >= 60 else f"{elapsed}s"}</span>'
                    f' &nbsp;—&nbsp; {msg}</div>',
                    unsafe_allow_html=True,
                )
            except Exception:
                pass
            self._stop_event.wait(1)

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2)


# ─────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────
DEFAULTS = {"result": None, "chat_history": [], "pipeline_done": False, "history": []}
for _k, _v in DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = list(_v) if isinstance(_v, list) else _v


def reset_to_input():
    st.session_state.result = None
    st.session_state.chat_history = []
    st.session_state.pipeline_done = False


# ─────────────────────────────────────────────────────────────────────────
# Small UI building blocks
# ─────────────────────────────────────────────────────────────────────────
def copy_button(text: str, label: str, key: str):
    """Clipboard copy via a tiny self-contained JS snippet — no new pip
    dependency. Uses st.iframe when available (the current API for this),
    falling back to the older components.html for older Streamlit installs.
    Runs in its own sandboxed iframe, so colors are hardcoded here rather
    than referencing the page's CSS variables."""
    safe_text = json.dumps(text or "")
    safe_key = re.sub(r"[^a-zA-Z0-9_]", "_", key)
    snippet = f"""
        <button id="{safe_key}" style="
            font-family: Inter, sans-serif; background:#191a20; color:#eeeeec;
            border:1px solid #26272e; border-radius:8px; padding:0.4rem 0.9rem;
            font-size:0.8rem; cursor:pointer;">{esc(label)}</button>
        <script>
        const btn_{safe_key} = document.getElementById('{safe_key}');
        btn_{safe_key}.addEventListener('click', function() {{
            navigator.clipboard.writeText({safe_text});
            btn_{safe_key}.innerText = 'Copied';
            setTimeout(() => btn_{safe_key}.innerText = {json.dumps(label)}, 1400);
        }});
        </script>
    """
    if hasattr(st, "iframe"):
        st.iframe(snippet, height=42)
    else:
        components.html(snippet, height=42)


def split_numbered_items(text: str) -> list:
    if not text or not text.strip():
        return []
    parts = re.split(r"\n(?=\s*\d+[\.\)]\s)", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts if len(parts) > 1 else [text.strip()]


def render_divider_list(text: str, empty_prefix: str):
    items = split_numbered_items(text)
    if not items or items[0].lower().startswith(empty_prefix.lower()):
        st.markdown(
            f'<div class="surface-panel"><span style="color:var(--text-faint);font-size:0.85rem">{esc(empty_prefix)}</span></div>',
            unsafe_allow_html=True,
        )
        return
    body = "".join(f'<div class="divider-item"><div class="divider-item-body">{esc(i)}</div></div>' for i in items)
    st.markdown(f'<div class="surface-panel">{body}</div>', unsafe_allow_html=True)


def build_markdown_report(result: dict) -> str:
    return (
        f"# {result.get('title', 'Meeting Summary')}\n\n"
        f"**Analyzed:** {result.get('analyzed_at', '')}  \n"
        f"**Source:** {result.get('source', '')}  \n"
        f"**Duration:** {format_duration(result.get('duration_seconds'))}\n\n"
        f"## Executive Summary\n{result.get('summary', '')}\n\n"
        f"## Action Items\n{result.get('action_items', '')}\n\n"
        f"## Key Decisions\n{result.get('key_decisions', '')}\n\n"
        f"## Open Questions\n{result.get('open_questions', '')}\n"
    )


def _pdf_safe(text: str) -> str:
    return (text or "").encode("latin-1", "replace").decode("latin-1")


def build_pdf_bytes(result: dict) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 9, _pdf_safe(result.get("title") or "Meeting Summary"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(130, 130, 130)
    meta = f"Generated {result.get('analyzed_at', '')}   |   Source: {result.get('source', '')}"
    pdf.multi_cell(0, 6, _pdf_safe(meta), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(20, 20, 20)
    pdf.ln(2)
    for heading, body in [
        ("Executive Summary", result.get("summary", "")),
        ("Action Items", result.get("action_items", "")),
        ("Key Decisions", result.get("key_decisions", "")),
        ("Open Questions", result.get("open_questions", "")),
    ]:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 7, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5.5, _pdf_safe(body) or "—", new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def save_uploaded_file(uploaded_file) -> str:
    suffix = "." + uploaded_file.name.split(".")[-1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


# ─────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-nav-title">AI Video Assistant</div>'
            '<div class="sidebar-nav-sub">Meeting intelligence workspace</div>',
            unsafe_allow_html=True,
        )

        if st.button("+ New Analysis", use_container_width=True):
            reset_to_input()
            st.rerun()

        st.markdown('<div class="sidebar-section-label">Recent Analyses</div>', unsafe_allow_html=True)
        if st.session_state.history:
            for i, item in enumerate(reversed(st.session_state.history)):
                label = item["title"][:30] + ("…" if len(item["title"]) > 30 else "")
                if st.button(label, key=f"hist_{i}", use_container_width=True):
                    st.session_state.result = item
                    st.session_state.chat_history = []
                    st.session_state.pipeline_done = True
                    st.rerun()
        else:
            st.markdown('<div class="history-empty">No analyses yet this session.</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section-label">Settings</div>', unsafe_allow_html=True)
        with st.expander("Environment status"):
            groq_ok = bool(os.getenv("GROQ_API_KEY"))
            sarvam_ok = bool(os.getenv("SARVAM_API_KEY"))
            st.markdown(f"- Groq API key: {'✅ detected' if groq_ok else '⚠️ missing'}")
            st.markdown(f"- Sarvam API key: {'✅ detected' if sarvam_ok else '— optional, not set'}")
            st.caption("Read from your environment / .env file. Nothing is stored by this app.")


# ─────────────────────────────────────────────────────────────────────────
# Top header bar
# ─────────────────────────────────────────────────────────────────────────
def render_top_header():
    st.markdown(
        f"""<div class="topbar fade-in">
            <div class="topbar-brand">
                <div class="topbar-logo">A</div>
                <div class="topbar-title">AI Video Assistant</div>
            </div>
            <a class="topbar-link" href="{GITHUB_URL}" target="_blank">⬚ GitHub</a>
        </div>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────
# Input workspace (empty state)
# ─────────────────────────────────────────────────────────────────────────
def render_hero():
    st.markdown('<div class="eyebrow fade-in">Meeting Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-h1 fade-in">Turn hours of video into useful information.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub fade-in">Transcribe meetings, extract decisions, and ask questions about your recordings.</div>',
        unsafe_allow_html=True,
    )
    st.write("")


def render_input_workspace():
    box = st.container(border=True)
    with box:
        mode = st.radio(
            "Input method", ["Paste YouTube URL", "Upload video/audio"],
            horizontal=True, label_visibility="collapsed",
        )
        st.write("")
        source_value, uploaded_file = None, None
        if mode == "Paste YouTube URL":
            source_value = st.text_input(
                "YouTube URL", placeholder="https://youtube.com/watch?v=...", label_visibility="collapsed"
            )
            st.caption(
                "If a YouTube link fails with a 403/Forbidden error, YouTube is blocking this "
                "server's IP — try again in a bit, or use Upload video/audio instead."
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload video or audio", type=SUPPORTED_UPLOAD_TYPES, label_visibility="collapsed"
            )
            st.caption("Supported formats: MP4, MP3, WAV, M4A")

        language_label = st.radio("Language", ["English", "Hinglish"], horizontal=True)
        language = "hinglish" if language_label == "Hinglish" else "english"

        st.write("")
        analyse_clicked = st.button("Analyse video", type="primary")
    return mode, source_value, uploaded_file, language, analyse_clicked


# ─────────────────────────────────────────────────────────────────────────
# Pipeline execution — identical backend call order to the original app.py
# ─────────────────────────────────────────────────────────────────────────
def render_checklist(step_states: dict, placeholder):
    rows = []
    for key, label in PIPELINE_STEPS:
        state = step_states.get(key, "pending")  # pending | active | done
        icon = {"done": "✓", "active": "◯", "pending": "○"}[state]
        rows.append(f'<div class="pl-row pl-{state}"><span class="pl-icon">{icon}</span><span class="pl-label">{label}</span></div>')
    placeholder.markdown('<div class="fade-in">' + "".join(rows) + "</div>", unsafe_allow_html=True)


def friendly_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if "403" in msg and ("forbidden" in low or "youtube" in low or "http error 403" in low):
        return (
            "YouTube blocked this server's request (HTTP 403 Forbidden). This happens when "
            "YouTube rate-limits or flags the server's IP address — it isn't something wrong "
            "with your link or this app's logic. Things that usually help:\n\n"
            "- Wait a minute and try again (YouTube's block is often temporary)\n"
            "- Use **Upload video/audio** instead and upload the file directly\n"
            "- If this keeps happening on a cloud deployment, the hosting provider's IP range "
            "may be blocked by YouTube more persistently — uploading a file avoids this entirely"
        )
    if "sarvam_api_key" in low or ("sarvam" in low and "key" in low):
        return "Sarvam API key is missing or invalid. Add `SARVAM_API_KEY` to your environment to use the Hinglish option."
    if "groq_api_key" in low or ("groq" in low and "key" in low):
        return "Groq API key is missing or invalid. Add `GROQ_API_KEY` to your environment."
    return f"Something went wrong: {msg}"


def run_pipeline(source: str, language: str):
    st.session_state.result = None
    st.session_state.chat_history = []
    start_time = datetime.now()

    step_states = {key: "pending" for key, _ in PIPELINE_STEPS}

    try:
        with st.status("Analysing your video…", expanded=True) as status:
            checklist_ph = st.empty()
            timer_ph = st.empty()
            render_checklist(step_states, checklist_ph)

            timer = LiveTimer(timer_ph)
            timer.start_timer()

            def mark(key, state):
                step_states[key] = state
                render_checklist(step_states, checklist_ph)

            try:
                mark("audio", "active")
                chunks = process_input(source)
                mark("audio", "done")

                # Real duration, computed from the chunk files audio_processor.py
                # already produced — reads existing files, doesn't touch that module.
                duration_seconds = 0.0
                try:
                    for c in chunks:
                        duration_seconds += len(AudioSegment.from_wav(c)) / 1000.0
                except Exception:
                    duration_seconds = 0.0

                mark("transcript", "active")
                transcript = transcribe_all(chunks, language)
                mark("transcript", "done")

                mark("title", "active")
                title = generate_title(transcript)
                mark("title", "done")

                mark("summary", "active")
                summary = summarize(transcript)
                mark("summary", "done")

                mark("extract", "active")
                action_items = extract_action_items(transcript)
                decisions = extract_key_decisions(transcript)
                questions = extract_questions(transcript)
                mark("extract", "done")

                mark("rag", "active")
                rag_chain = build_rag_chain(transcript)
                mark("rag", "done")
            finally:
                timer.stop()
                timer_ph.empty()

            status.update(label="Analysis complete", state="complete", expanded=False)

        elapsed = (datetime.now() - start_time).total_seconds()
        result = {
            "title": title,
            "transcript": transcript,
            "summary": summary,
            "action_items": action_items,
            "key_decisions": decisions,
            "open_questions": questions,
            "rag_chain": rag_chain,
            "source": source,
            "language": language,
            "duration_seconds": duration_seconds,
            "analyzed_at": datetime.now().strftime("%d %b %Y, %H:%M"),
            "processing_seconds": elapsed,
        }
        st.session_state.result = result
        st.session_state.history.append(result)
        st.session_state.pipeline_done = True
        st.rerun()

    except Exception as e:  # noqa: BLE001
        st.error(friendly_error(e))


# ─────────────────────────────────────────────────────────────────────────
# Results workspace
# ─────────────────────────────────────────────────────────────────────────
def render_results_header(result: dict):
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="eyebrow">Session Title</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hero-h1" style="font-size:1.7rem">{esc(result["title"])}</div>', unsafe_allow_html=True)
        engine_label = "Whisper" if result.get("language") == "english" else "Sarvam AI"
        lang_label = "English" if result.get("language") == "english" else "Hinglish → English"
        st.markdown(
            f'<div class="meta-line">{format_duration(result.get("duration_seconds"))} · {esc(lang_label)} · {esc(engine_label)}</div>',
            unsafe_allow_html=True,
        )
    with col2:
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.download_button("Export .md", data=build_markdown_report(result),
                                file_name="meeting-report.md", mime="text/markdown", use_container_width=True)
        with b2:
            st.download_button("Export PDF", data=build_pdf_bytes(result),
                                file_name="meeting-report.pdf", mime="application/pdf", use_container_width=True)
        with b3:
            copy_button(build_markdown_report(result), "Copy notes", key="copy_notes")
        with b4:
            if st.button("New Analysis", use_container_width=True):
                reset_to_input()
                st.rerun()

    engine_label = "Whisper (English)" if result.get("language") == "english" else "Sarvam AI (Hinglish → English)"
    st.markdown(
        f"""<div class="meta-row">
            <div class="meta-chip"><span class="meta-label">Duration</span><span class="meta-value">{format_duration(result.get('duration_seconds'))}</span></div>
            <div class="meta-chip"><span class="meta-label">Analyzed</span><span class="meta-value">{esc(result.get('analyzed_at', '—'))}</span></div>
            <div class="meta-chip"><span class="meta-label">Engine</span><span class="meta-value">{esc(engine_label)}</span></div>
            <div class="meta-chip"><span class="meta-label">Processing time</span><span class="meta-value">{format_duration(result.get('processing_seconds'))}</span></div>
            <div class="meta-chip"><span class="meta-label">Source</span><span class="meta-value" style="max-width:16rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;display:inline-block">{esc(result.get('source', ''))}</span></div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("---")


def render_summary_section(result: dict):
    st.markdown('<div class="section-label">Executive Summary</div>', unsafe_allow_html=True)
    body = esc(result.get("summary", "")).replace("\n", "<br>")
    st.markdown(f'<div class="surface-panel fade-in"><div class="divider-item-body">{body}</div></div>', unsafe_allow_html=True)
    st.write("")


ACTION_ITEM_PATTERN = re.compile(
    r"task\s*[:\-]\s*(.+?)(?=\n\s*owner\s*[:\-]|\n\s*deadline\s*[:\-]|$)"
    r"(?:\n\s*owner\s*[:\-]\s*(.+?))?(?=\n\s*deadline\s*[:\-]|$)"
    r"(?:\n\s*deadline\s*[:\-]\s*(.+?))?$",
    re.IGNORECASE | re.DOTALL,
)


def parse_action_items(text: str):
    """Best-effort parse of 'Task / Owner / Deadline' items (the shape
    core/extractor.py's prompt already asks for) into structured rows for a
    real table. Returns None if fewer than half the items parse cleanly, so
    the caller can fall back to the plain text rendering instead of showing
    a half-empty, unconvincing table."""
    items = split_numbered_items(text)
    if not items or items[0].lower().startswith("no action items"):
        return []
    rows = []
    parsed_count = 0
    for item in items:
        clean = re.sub(r"^\s*\d+[\.\)]\s*", "", item).strip()
        m = ACTION_ITEM_PATTERN.search(clean)
        if m and m.group(1):
            task = m.group(1).strip().rstrip(".")
            owner = (m.group(2) or "Not specified").strip().rstrip(".")
            deadline = (m.group(3) or "Not specified").strip().rstrip(".")
            rows.append({"task": task, "owner": owner, "deadline": deadline})
            parsed_count += 1
        else:
            rows.append({"task": clean, "owner": "—", "deadline": "—"})
    if parsed_count < max(1, len(items) // 2):
        return None
    return rows


def render_action_items_table(text: str):
    rows = parse_action_items(text)
    if rows is None:
        render_divider_list(text, "No action items found.")
        return
    if not rows:
        st.markdown(
            '<div class="surface-panel"><span style="color:var(--text-faint);font-size:0.85rem">No action items found.</span></div>',
            unsafe_allow_html=True,
        )
        return
    body_rows = "".join(
        f'<tr><td class="out-task">{esc(r["task"])}</td><td>{esc(r["owner"])}</td><td class="out-due">{esc(r["deadline"])}</td></tr>'
        for r in rows
    )
    st.markdown(
        f"""<div class="surface-panel fade-in">
            <table class="out-table">
                <thead><tr><th>Task</th><th>Owner</th><th>Due</th></tr></thead>
                <tbody>{body_rows}</tbody>
            </table>
        </div>""",
        unsafe_allow_html=True,
    )


def render_bullet_list(text: str, empty_label: str, dot_class: str = ""):
    items = split_numbered_items(text)
    if not items or items[0].lower().startswith(empty_label.lower().split(" found")[0]):
        st.markdown(f'<span style="color:var(--text-faint);font-size:0.85rem">{esc(empty_label)}</span>', unsafe_allow_html=True)
        return
    lis = "".join(
        f'<li><span class="bullet-dot {dot_class}"></span><span>{esc(re.sub(r"^\\s*\\d+[.)]\\s*", "", i))}</span></li>'
        for i in items
    )
    st.markdown(f'<ul class="bullet-list fade-in">{lis}</ul>', unsafe_allow_html=True)


def render_insights_section(result: dict):
    st.markdown('<div class="section-label">Key Insights</div>', unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2], gap="medium")
    with col_left:
        st.markdown('<div style="font-size:0.8rem;font-weight:600;color:var(--text-muted);margin-bottom:0.4rem">Action Items</div>', unsafe_allow_html=True)
        render_action_items_table(result.get("action_items", ""))
    with col_right:
        st.markdown('<div style="font-size:0.8rem;font-weight:600;color:var(--text-muted);margin-bottom:0.4rem">Key Decisions</div>', unsafe_allow_html=True)
        render_bullet_list(result.get("key_decisions", ""), "No key decisions found.")
        st.write("")
        st.markdown('<div style="font-size:0.8rem;font-weight:600;color:var(--text-muted);margin-bottom:0.4rem">Open Questions</div>', unsafe_allow_html=True)
        render_bullet_list(result.get("open_questions", ""), "No open questions found.", dot_class="amber")
    st.write("")


def render_transcript_section(result: dict):
    st.markdown('<div class="section-label">Transcript</div>', unsafe_allow_html=True)
    box = st.container(border=True)
    with box:
        top1, top2 = st.columns([3, 1])
        with top1:
            query = st.text_input("Search transcript", placeholder="Search the transcript…", label_visibility="collapsed")
        with top2:
            st.download_button("Download .txt", data=result.get("transcript", ""),
                                file_name="transcript.txt", mime="text/plain", use_container_width=True)

        transcript = result.get("transcript", "")
        marked = transcript
        if query.strip():
            pattern = re.compile(re.escape(query.strip()), re.IGNORECASE)
            marked = pattern.sub(lambda m: f"⦃{m.group(0)}⦄", transcript)
            match_count = len(pattern.findall(transcript))
            st.caption(f"{match_count} match{'es' if match_count != 1 else ''} for “{query.strip()}”")

        safe = esc(marked).replace("⦃", "<mark>").replace("⦄", "</mark>")
        st.markdown(
            f'<div style="max-height:320px;overflow-y:auto;font-size:0.85rem;line-height:1.7;'
            f'color:var(--text-muted);white-space:pre-wrap;padding-right:0.4rem">{safe}</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        copy_button(transcript, "Copy transcript", key="copy_transcript")
        st.caption(
            "Speaker labels and clickable timestamps aren't available yet — the current "
            "transcription pipeline produces one continuous transcript with no per-segment "
            "timing or diarization."
        )
    st.write("")


def ask(result: dict, question: str):
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.spinner("Thinking…"):
        try:
            answer = ask_question(result["rag_chain"], question)
        except Exception as e:  # noqa: BLE001
            answer = f"Something went wrong answering that: {e}"
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "note": "Grounded in this meeting's transcript via retrieval-augmented search.",
    })


def render_chat_section(result: dict):
    st.markdown('<div class="eyebrow">Ask your meeting</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub" style="margin-bottom:0.9rem">Search your transcript using natural language.</div>', unsafe_allow_html=True)

    box = st.container(border=True)
    with box:
        if not st.session_state.chat_history:
            st.caption("Try asking:")
            row1 = st.columns(3)
            for i, q in enumerate(SUGGESTED_QUESTIONS[:3]):
                with row1[i]:
                    if st.button(q, key=f"sugg_{i}", use_container_width=True):
                        ask(result, q)
                        st.rerun()
            row2 = st.columns(2)
            for i, q in enumerate(SUGGESTED_QUESTIONS[3:]):
                with row2[i]:
                    if st.button(q, key=f"sugg_{i + 3}", use_container_width=True):
                        ask(result, q)
                        st.rerun()
        else:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    if msg["role"] == "assistant" and msg.get("note"):
                        st.caption(msg["note"])

        question = st.chat_input("Ask a question about this meeting…")
        if question and question.strip():
            ask(result, question.strip())
            st.rerun()

        if st.session_state.chat_history:
            if st.button("Clear conversation"):
                st.session_state.chat_history = []
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
inject_css()
render_sidebar()
render_top_header()

if st.session_state.result:
    _result = st.session_state.result
    render_results_header(_result)
    left, right = st.columns([3, 2], gap="large")
    with left:
        render_summary_section(_result)
        render_insights_section(_result)
    with right:
        render_transcript_section(_result)
    st.markdown("---")
    render_chat_section(_result)
else:
    render_hero()
    _mode, _source_value, _uploaded_file, _language, _analyse_clicked = render_input_workspace()
    if _analyse_clicked:
        if _mode == "Paste YouTube URL":
            if not (_source_value or "").strip():
                st.error("Please paste a YouTube URL.")
            else:
                run_pipeline(_source_value.strip(), _language)
        else:
            if _uploaded_file is None:
                st.error("Please upload a video or audio file.")
            else:
                _temp_path = save_uploaded_file(_uploaded_file)
                run_pipeline(_temp_path, _language)
