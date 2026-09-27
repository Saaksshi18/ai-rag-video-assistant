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
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from fpdf import FPDF
from pydub import AudioSegment

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

SUPPORTED_UPLOAD_TYPES = ["mp4", "mp3", "wav", "m4a"]
SUGGESTED_QUESTIONS = [
    "What decisions were made?",
    "What are my action items?",
    "When is the project deadline?",
    "What concerns were raised?",
    "Summarise the deployment discussion.",
]

# ─────────────────────────────────────────────────────────────────────────
# Design system — restrained dark theme, Inter type, subtle borders only
# ─────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --bg: #0c0c0f;
        --surface: #131318;
        --surface-2: #191a20;
        --surface-3: #202127;
        --border: #26272e;
        --border-soft: #1d1e24;
        --accent: #7c6ef2;
        --accent-soft: rgba(124, 110, 242, 0.14);
        --accent-2: #4fb8c9;
        --accent-2-soft: rgba(79, 184, 201, 0.14);
        --text: #eeeeec;
        --text-muted: #96969f;
        --text-faint: #5c5c66;
        --success: #4caf82;
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
    button[kind="primary"] { background: var(--accent) !important; border: 1px solid var(--accent) !important; color: #fff !important; }
    button[kind="primary"]:hover { background: #6f61e8 !important; border-color: #6f61e8 !important; }

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
    .hero-h1 { font-size: clamp(1.7rem, 3.4vw, 2.4rem); font-weight: 700; line-height: 1.18; color: var(--text); margin: 0 0 0.5rem 0; }
    .hero-sub { color: var(--text-muted); font-size: 0.95rem; max-width: 34rem; line-height: 1.55; }

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

    @media (max-width: 640px) {
        .hero-h1 { font-size: 1.5rem; }
        .surface-panel { padding: 0.9rem; }
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
# Input workspace (empty state)
# ─────────────────────────────────────────────────────────────────────────
def render_hero():
    st.markdown('<div class="eyebrow">Meeting Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-h1">Turn hours of video into useful information.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Transcribe meetings, extract decisions, and ask questions about your recordings.</div>',
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
def run_pipeline(source: str, language: str):
    st.session_state.result = None
    st.session_state.chat_history = []
    start_time = datetime.now()

    try:
        with st.status("Analysing your video…", expanded=True) as status:
            status.write("01 · Audio Processing")
            chunks = process_input(source)

            # Real duration, computed from the chunk files audio_processor.py
            # already produced — reads existing files, doesn't touch that module.
            duration_seconds = 0.0
            try:
                for c in chunks:
                    duration_seconds += len(AudioSegment.from_wav(c)) / 1000.0
            except Exception:
                duration_seconds = 0.0

            status.write("02 · Transcription")
            transcript = transcribe_all(chunks, language)

            status.write("03 · Summary")
            title = generate_title(transcript)
            summary = summarize(transcript)

            status.write("04 · Insights")
            action_items = extract_action_items(transcript)
            decisions = extract_key_decisions(transcript)
            questions = extract_questions(transcript)

            status.write("05 · RAG Indexing")
            rag_chain = build_rag_chain(transcript)

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
        st.error(f"Something went wrong: {e}")


# ─────────────────────────────────────────────────────────────────────────
# Results workspace
# ─────────────────────────────────────────────────────────────────────────
def render_results_header(result: dict):
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="eyebrow">Session Title</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hero-h1" style="font-size:1.5rem">{esc(result["title"])}</div>', unsafe_allow_html=True)
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
    st.markdown(f'<div class="surface-panel"><div class="divider-item-body">{body}</div></div>', unsafe_allow_html=True)
    st.write("")


def render_insights_section(result: dict):
    st.markdown('<div class="section-label">Key Insights</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Action Items", "Key Decisions", "Open Questions"])
    with tab1:
        render_divider_list(result.get("action_items", ""), "No action items found.")
    with tab2:
        render_divider_list(result.get("key_decisions", ""), "No key decisions found.")
    with tab3:
        render_divider_list(result.get("open_questions", ""), "No open questions found.")
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
            marked = pattern.sub(lambda m: f"\u2983{m.group(0)}\u2984", transcript)
            match_count = len(pattern.findall(transcript))
            st.caption(f"{match_count} match{'es' if match_count != 1 else ''} for \u201c{query.strip()}\u201d")

        safe = esc(marked).replace("\u2983", "<mark>").replace("\u2984", "</mark>")
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
