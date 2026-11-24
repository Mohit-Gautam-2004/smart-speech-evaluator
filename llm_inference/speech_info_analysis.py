# ==========================================================
# Imports
# ==========================================================

import datetime
import json
import os
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import requests
import soundfile as sf
import streamlit as st
import torch
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from transformers import pipeline

try:
    from db_manager import (
        delete_reference_topic,
        delete_speech_sections,
        get_latest_attempt,
        list_practice_attempts,
        list_reference_topics,
        list_speech_sections,
        save_practice_attempt,
        save_reference_topic,
        save_speech_section,
        test_connection,
    )
except ImportError:
    from llm_inference.db_manager import (
        delete_reference_topic,
        delete_speech_sections,
        get_latest_attempt,
        list_practice_attempts,
        list_reference_topics,
        list_speech_sections,
        save_practice_attempt,
        save_reference_topic,
        save_speech_section,
        test_connection,
    )


# ==========================================================
# Page Config
# ==========================================================

st.set_page_config(page_title="SmartSpeak — AI Speech Evaluator", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")

# ==========================================================
# Custom CSS — Professional Dark-Gradient Theme
# ==========================================================
st.markdown("""
<style>
/* ── Google Fonts ───────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ─────────────────────────────────────── */
:root {
  --bg-primary:     #0d1117;
  --bg-secondary:   #161b22;
  --bg-card:        #1c2333;
  --bg-card-hover:  #21293a;
  --accent-blue:    #58a6ff;
  --accent-purple:  #bc8cff;
  --accent-green:   #3fb950;
  --accent-orange:  #f0883e;
  --accent-red:     #f85149;
  --accent-cyan:    #39d0d8;
  --text-primary:   #e6edf3;
  --text-secondary: #8b949e;
  --text-muted:     #484f58;
  --border:         #30363d;
  --border-active:  #58a6ff;
  --radius:         12px;
  --radius-sm:      8px;
  --shadow:         0 8px 32px rgba(0,0,0,0.4);
  --shadow-sm:      0 2px 8px rgba(0,0,0,0.3);
  --gradient-blue:  linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #1565c0 100%);
  --gradient-hero:  linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1c2333 100%);
  --gradient-card:  linear-gradient(145deg, #1c2333 0%, #21293a 100%);
}

/* ── App shell ──────────────────────────────────────────── */
.stApp {
  background: var(--bg-primary) !important;
  font-family: 'Inter', sans-serif !important;
  color: var(--text-primary) !important;
}

/* ── Hide default streamlit chrome ──────────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Main content padding ───────────────────────────────── */
.block-container {
  padding: 1.5rem 2.5rem 3rem 2.5rem !important;
  max-width: 1200px;
}

/* ── Sidebar ────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0d1117 0%, #161b22 100%) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
  color: var(--text-primary) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  color: var(--accent-blue) !important;
}
[data-testid="stSidebarContent"] {
  padding: 1.5rem 1rem !important;
}

/* ── All text ───────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 { color: var(--text-primary) !important; }
p, span, div, label { color: var(--text-secondary) !important; }

/* ── Buttons ────────────────────────────────────────────── */
.stButton > button {
  background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  padding: 0.55rem 1.4rem !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.875rem !important;
  letter-spacing: 0.02em !important;
  box-shadow: 0 4px 15px rgba(31, 111, 235, 0.3) !important;
  transition: all 0.2s ease !important;
  cursor: pointer !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #388bfd 0%, #58a6ff 100%) !important;
  box-shadow: 0 6px 20px rgba(88, 166, 255, 0.4) !important;
  transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
  background: linear-gradient(135deg, #21293a 0%, #2d3748 100%) !important;
  border: 1px solid var(--border) !important;
  box-shadow: none !important;
}
.stButton > button[kind="secondary"]:hover {
  border-color: var(--accent-blue) !important;
  box-shadow: 0 0 0 1px var(--accent-blue) !important;
}

/* ── Inputs & Textareas ─────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div,
.stMultiSelect > div > div {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.9rem !important;
  transition: border-color 0.2s ease !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  border-color: var(--accent-blue) !important;
  box-shadow: 0 0 0 3px rgba(88,166,255,0.15) !important;
}

/* ── Selectbox / Radio ──────────────────────────────────── */
.stSelectbox label, .stRadio label, .stTextInput label, .stTextArea label {
  color: var(--text-primary) !important;
  font-weight: 500 !important;
  font-size: 0.875rem !important;
}
[data-testid="stSelectboxVirtualDropdown"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
}
.stRadio > div { gap: 0.5rem !important; }
.stRadio > div > label {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  padding: 0.4rem 1rem !important;
  transition: all 0.2s ease !important;
}
.stRadio > div > label:hover { border-color: var(--accent-blue) !important; }

/* ── Progress bar ───────────────────────────────────────── */
.stProgress > div > div > div > div {
  background: linear-gradient(90deg, #1f6feb, #58a6ff, #bc8cff) !important;
  border-radius: 100px !important;
}
.stProgress > div > div > div {
  background: var(--bg-card) !important;
  border-radius: 100px !important;
  height: 8px !important;
}

/* ── Metrics ────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 1rem 1.2rem !important;
  box-shadow: var(--shadow-sm) !important;
  transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stMetric"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: var(--shadow) !important;
  border-color: var(--accent-blue) !important;
}
[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; font-size: 0.8rem !important; font-weight: 500 !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; font-size: 1.6rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.85rem !important; font-weight: 600 !important; }

/* ── Expanders ──────────────────────────────────────────── */
details {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 0.5rem !important;
  margin-bottom: 0.75rem !important;
}
details summary {
  color: var(--text-primary) !important;
  font-weight: 600 !important;
  cursor: pointer !important;
  padding: 0.5rem !important;
}
details[open] { border-color: var(--accent-blue) !important; }

/* ── Success / Warning / Error / Info banners ───────────── */
.stSuccess {
  background: rgba(63,185,80,0.12) !important;
  border: 1px solid rgba(63,185,80,0.4) !important;
  border-radius: var(--radius-sm) !important;
  color: #3fb950 !important;
}
.stWarning {
  background: rgba(240,136,62,0.12) !important;
  border: 1px solid rgba(240,136,62,0.4) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--accent-orange) !important;
}
.stError {
  background: rgba(248,81,73,0.12) !important;
  border: 1px solid rgba(248,81,73,0.4) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--accent-red) !important;
}
.stInfo {
  background: rgba(88,166,255,0.08) !important;
  border: 1px solid rgba(88,166,255,0.3) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--accent-blue) !important;
}

/* ── Dataframe ──────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  overflow: hidden !important;
}
.stDataFrame thead th {
  background: var(--bg-secondary) !important;
  color: var(--accent-blue) !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.05em !important;
}
.stDataFrame tbody tr { background: var(--bg-card) !important; }
.stDataFrame tbody tr:hover { background: var(--bg-card-hover) !important; }

/* ── Audio widget ───────────────────────────────────────── */
audio {
  width: 100% !important;
  border-radius: var(--radius-sm) !important;
  accent-color: var(--accent-blue) !important;
}

/* ── Status widget ──────────────────────────────────────── */
[data-testid="stStatusWidget"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}

/* ── Download button ────────────────────────────────────── */
.stDownloadButton > button {
  background: linear-gradient(135deg, #2d6a4f 0%, #40916c 100%) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 600 !important;
  box-shadow: 0 4px 15px rgba(63,185,80,0.25) !important;
}
.stDownloadButton > button:hover {
  background: linear-gradient(135deg, #40916c 0%, #52b788 100%) !important;
  box-shadow: 0 6px 20px rgba(63,185,80,0.4) !important;
}

/* ── Caption ────────────────────────────────────────────── */
.stCaption { color: var(--text-muted) !important; font-size: 0.78rem !important; }

/* ── Divider ────────────────────────────────────────────── */
hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  margin: 1.5rem 0 !important;
}

/* ── Scrollbar ──────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
</style>
""", unsafe_allow_html=True)


# ==========================================================
# Load .env from project root
# ==========================================================

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMOTION_SERVICE_URL = os.getenv("EMOTION_SERVICE_URL", "http://127.0.0.1:5001")


# ==========================================================
# Constants
# ==========================================================

RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

STEPS = {
    1: "Reference Setup",
    2: "Audio Input",
    3: "Transcription",
    4: "RAG Retrieval",
    5: "Detailed Scoring",
}


# ==========================================================
# Cached Model Loaders
# ==========================================================

@st.cache_resource
def load_stt_model():
    """Load Whisper once for repeated transcriptions."""
    return pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-base",
        device=0 if torch.cuda.is_available() else -1,
        chunk_length_s=30,
        stride_length_s=5,
    )


@st.cache_resource
def load_embedding_model():
    """Open-source sentence-transformer model for embeddings."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_wav2vec2_emotion_model():
    """Load pre-trained Wav2Vec 2.0 speech emotion recognition model."""
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, pipeline

    try:
        fe = AutoFeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er", local_files_only=True)
        m = AutoModelForAudioClassification.from_pretrained("superb/wav2vec2-base-superb-er", local_files_only=True)
    except Exception:
        fe = AutoFeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")
        m = AutoModelForAudioClassification.from_pretrained("superb/wav2vec2-base-superb-er")

    device_idx = 0 if torch.cuda.is_available() else -1
    return pipeline(
        task="audio-classification",
        model=m,
        feature_extractor=fe,
        device=device_idx,
    )


def predict_emotion_wav2vec2_local(audio_path: Path) -> dict[str, Any]:
    """In-process Wav2Vec 2.0 inference fallback if emotion backend is not reachable."""
    import librosa

    audio_data, sample_rate = sf.read(str(audio_path), dtype="float32")
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    if sample_rate != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    classifier = load_wav2vec2_emotion_model()
    raw_results = classifier({"array": audio_data, "sampling_rate": sample_rate})

    label_map = {"neu": "Neutral", "hap": "Happy", "ang": "Angry", "sad": "Sad"}
    prob_map = {"Neutral": 0.0, "Angry": 0.0, "Happy": 0.0, "Sad": 0.0}
    for item in raw_results:
        key = label_map.get(item["label"], item["label"])
        if key in prob_map:
            prob_map[key] = float(item["score"])

    top_label = max(prob_map.items(), key=lambda x: x[1])[0]
    labels_order = ["Neutral", "Angry", "Happy", "Sad"]
    return {
        "emotion_index": labels_order.index(top_label),
        "emotion_label": top_label,
        "confidence": prob_map[top_label],
        "probabilities": prob_map,
        "model": "wav2vec2 (transformer)",
        "audio_file": str(audio_path),
    }


# ==========================================================
# Helpers
# ==========================================================

def save_audio_bytes(raw_bytes: bytes, suffix: str = ".wav") -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = RECORDINGS_DIR / f"recording_{timestamp}{suffix}"
    with open(file_path, "wb") as file_handle:
        file_handle.write(raw_bytes)
    return file_path


def load_audio_bytes(file_path: Path) -> bytes:
    with open(file_path, "rb") as file_handle:
        return file_handle.read()


def read_text_file(uploaded_file) -> str:
    raw = uploaded_file.getvalue()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="ignore")


def speech_to_text(file_path: Path) -> str:
    stt = load_stt_model()
    audio_data, sample_rate = sf.read(str(file_path), dtype="float32")
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    result = stt({"array": audio_data, "sampling_rate": sample_rate})
    return str(result.get("text", "")).strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def build_faiss_index(chunks: list[str]) -> tuple[faiss.IndexFlatIP, np.ndarray]:
    model = load_embedding_model()
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings


def retrieve_context(query: str, index: faiss.IndexFlatIP, chunks: list[str], k: int = 4) -> list[dict[str, Any]]:
    if not query.strip() or not chunks:
        return []

    model = load_embedding_model()
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)

    k = min(k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []
    for score, chunk_idx in zip(scores[0], indices[0]):
        if chunk_idx == -1:
            continue
        results.append(
            {
                "chunk_id": int(chunk_idx),
                "similarity": float(score),
                "text": chunks[int(chunk_idx)],
            }
        )
    return results


def extract_json_from_response(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```json"):
        candidate = candidate[7:]
    if candidate.startswith("```"):
        candidate = candidate[3:]
    if candidate.endswith("```"):
        candidate = candidate[:-3]

    candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def tokenize_for_scoring(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z]{3,}", text.lower())}


def local_scoring(expected: str, transcript: str, retrieved: list[dict[str, Any]]) -> dict[str, Any]:
    expected_tokens = tokenize_for_scoring(expected)
    transcript_tokens = tokenize_for_scoring(transcript)

    if expected_tokens:
        overlap_ratio = len(expected_tokens & transcript_tokens) / len(expected_tokens)
    else:
        overlap_ratio = 0.0

    retrieved_text = " ".join(item["text"] for item in retrieved)
    retrieved_tokens = tokenize_for_scoring(retrieved_text)
    rag_alignment = 0.0
    if retrieved_tokens:
        rag_alignment = len(retrieved_tokens & transcript_tokens) / len(retrieved_tokens)

    relevance = int(min(100, max(0, round(overlap_ratio * 100))))
    coverage = int(min(100, max(0, round((0.6 * overlap_ratio + 0.4 * rag_alignment) * 100))))
    accuracy = int(min(100, max(0, round((0.5 * overlap_ratio + 0.5 * rag_alignment) * 100))))
    clarity = int(min(100, max(30, min(95, len(transcript.split()) * 2))))
    overall = int(round((relevance * 0.35) + (coverage * 0.25) + (accuracy * 0.25) + (clarity * 0.15)))

    return {
        "overall_score": overall,
        "scores": {
            "topic_relevance": relevance,
            "key_point_coverage": coverage,
            "content_accuracy": accuracy,
            "clarity": clarity,
        },
        "step_by_step_analysis": [
            {
                "step": "Topic Relevance",
                "score": relevance,
                "analysis": "Token overlap between expected content and transcript.",
            },
            {
                "step": "Key Point Coverage",
                "score": coverage,
                "analysis": "Checks how many expected ideas are reflected with RAG context support.",
            },
            {
                "step": "Content Accuracy",
                "score": accuracy,
                "analysis": "Measures alignment between transcript and retrieved knowledge chunks.",
            },
            {
                "step": "Clarity",
                "score": clarity,
                "analysis": "Estimated from transcript length and completeness.",
            },
        ],
        "strengths": ["Contains relevant terms from expected content."] if relevance >= 70 else ["Transcript captured basic topic terms."],
        "improvements": ["Add more specific key points from expected content."] if coverage < 70 else ["Improve precision for edge details."],
        "final_summary": "Generated via local fallback scorer (Gemini unavailable).",
    }


def get_wav_for_emotion(audio_path: Path) -> Path:
    """Ensure emotion backend receives a wav path usable by scipy wav reader."""
    if audio_path.suffix.lower() == ".wav":
        return audio_path

    converted_path = audio_path.with_name(f"{audio_path.stem}_emotion.wav")
    audio_data, sample_rate = sf.read(str(audio_path), dtype="float32")
    sf.write(str(converted_path), audio_data, sample_rate, subtype="PCM_16")
    return converted_path


# ------------------------------------------------------------------
# Gemini multimodal tone predictor
# ------------------------------------------------------------------

_TONE_PROMPT = (
    "Listen to the audio clip(s) provided. "
    "Classify the overall emotional tone as exactly ONE of: Neutral, Happy, Angry, Sad. "
    "Return ONLY valid JSON — no markdown, no extra text — in this exact schema:\n"
    '{"emotion_label": "<one of Neutral/Happy/Angry/Sad>", '
    '"confidence": <float 0.0-1.0>, '
    '"probabilities": {"Neutral": <float>, "Happy": <float>, "Angry": <float>, "Sad": <float>}, '
    '"reasoning": "<one sentence>"}'
)


def predict_tone_gemini(audio_paths: list[Path]) -> dict[str, Any]:
    """Send one OR multiple audio files to Gemini in a SINGLE API call.

    Limit usage: one call per evaluation regardless of section count.
    Falls back gracefully if the key is missing or the call fails.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured.")

    try:
        from google import genai as _genai
        from google.genai import types as _gtypes
    except ImportError:
        raise RuntimeError("google-genai SDK not installed (pip install google-genai).")

    client = _genai.Client(api_key=GEMINI_API_KEY)

    # Build parts list: one audio blob per section + the instruction prompt
    parts: list[Any] = []
    for ap in audio_paths:
        wav_path = get_wav_for_emotion(ap)
        audio_bytes = wav_path.read_bytes()
        parts.append(_gtypes.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"))

    if len(audio_paths) > 1:
        preamble = (
            f"There are {len(audio_paths)} audio clips representing different sections "
            "of a speech (Introduction, Body, Conclusion, etc.). "
            "Analyse them together and give ONE overall tone classification. "
        )
        parts.insert(0, _gtypes.Part.from_text(text=preamble))

    parts.append(_gtypes.Part.from_text(text=_TONE_PROMPT))

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",        # cheapest multimodal model, fast
        contents=[_gtypes.Content(parts=parts, role="user")],
        config=_gtypes.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=200,       # keep output tiny → saves quota
        ),
    )

    raw = response.text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()

    parsed = json.loads(raw)

    label = str(parsed.get("emotion_label", "Neutral")).strip().title()
    # Clamp label to known set
    if label not in ("Neutral", "Happy", "Angry", "Sad"):
        label = "Neutral"
    confidence = float(parsed.get("confidence", 0.8))
    probs = parsed.get("probabilities", {})
    # Ensure all keys present
    for k in ("Neutral", "Happy", "Angry", "Sad"):
        probs.setdefault(k, 0.0)

    labels_order = ["Neutral", "Angry", "Happy", "Sad"]
    return {
        "emotion_index": labels_order.index(label) if label in labels_order else 0,
        "emotion_label": label,
        "confidence": confidence,
        "probabilities": probs,
        "model": "gemini-2.0-flash (multimodal)",
        "reasoning": parsed.get("reasoning", ""),
        "audio_file": str(audio_paths[0]),
    }


def call_emotion_service(
    audio_path: Path,
    model_name: str = "wav2vec2",
    timeout_sec: int = 60,
) -> dict[str, Any]:
    """Single-file emotion service (legacy / wav2vec2 path)."""
    try:
        payload = {"audio_path": str(audio_path)}
        params = {"model": model_name, "mfcc_len": "39"}
        response = requests.post(
            f"{EMOTION_SERVICE_URL.rstrip('/')}/predict",
            json=payload,
            params=params,
            timeout=timeout_sec,
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass

    # Fall back to in-process Wav2Vec 2.0 pipeline if backend is unavailable or wav2vec2 is requested
    if model_name == "wav2vec2":
        return predict_emotion_wav2vec2_local(audio_path)
    raise RuntimeError(f"Emotion backend at {EMOTION_SERVICE_URL} is unreachable for model '{model_name}'.")


def tone_score_from_result(predicted_tone: str, confidence: float, expected_tone: str) -> int:
    expected = (expected_tone or "Any").strip().title()
    predicted = (predicted_tone or "").strip().title()
    confidence = float(max(0.0, min(1.0, confidence)))

    if expected == "Any":
        return int(round(50 + (confidence * 50)))
    if predicted == expected:
        return int(round(70 + (confidence * 30)))
    return int(round((1.0 - confidence) * 30))


def merge_context_and_tone(context_score: int, tone_score: int, tone_weight: float = 0.25) -> int:
    context_weight = 1.0 - tone_weight
    merged = (context_score * context_weight) + (tone_score * tone_weight)
    return int(round(max(0.0, min(100.0, merged))))


def llm_scoring(expected: str, transcript: str, retrieved: list[dict[str, Any]]) -> dict[str, Any]:
    if not GEMINI_API_KEY:
        return local_scoring(expected, transcript, retrieved)

    rag_context = "\n\n".join(
        [f"[Chunk {item['chunk_id']} | sim={item['similarity']:.3f}] {item['text']}" for item in retrieved]
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
        google_api_key=GEMINI_API_KEY,
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a strict speech evaluator using RAG evidence.

Expected Content:
{expected}

Spoken Transcript:
{spoken}

Retrieved RAG Evidence:
{rag_context}

Evaluate in a step-by-step manner and return JSON only.

Required JSON schema:
{{
  "overall_score": <int 0-100>,
  "scores": {{
    "topic_relevance": <int 0-100>,
    "key_point_coverage": <int 0-100>,
    "content_accuracy": <int 0-100>,
    "clarity": <int 0-100>
  }},
  "step_by_step_analysis": [
    {{"step": "Topic Relevance", "score": <int>, "analysis": "..."}},
    {{"step": "Key Point Coverage", "score": <int>, "analysis": "..."}},
    {{"step": "Content Accuracy", "score": <int>, "analysis": "..."}},
    {{"step": "Clarity", "score": <int>, "analysis": "..."}}
  ],
  "strengths": ["...", "..."],
  "improvements": ["...", "..."],
  "final_summary": "..."
}}

Rules:
- Scores must be integers between 0 and 100.
- Be concise and evidence-driven.
- Do not add any text outside JSON.
"""
    )

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke(
        {
            "expected": expected.strip(),
            "spoken": transcript.strip(),
            "rag_context": rag_context.strip(),
        }
    )
    return extract_json_from_response(response)


DEFAULT_EXPECTED_CONTENT = (
    "Good morning everyone. Today I want to talk about the importance of artificial intelligence "
    "in education. AI makes learning more personalized, accessible, and efficient. However, technology "
    "should support teachers rather than replace them. Thank you for listening."
)

DEFAULT_KNOWLEDGE_TEXT = (
    "Artificial Intelligence is transforming education through personalized learning, automated assessments, "
    "and intelligent tutoring systems. It helps students understand difficult concepts at their own pace and "
    "provides instant feedback. However, AI should complement human teachers, protect student privacy, and "
    "ensure equal access to education."
)


def initialize_state() -> None:
    defaults = {
        "current_step": 1,
        "user_id": "speaker_1",
        "topic_name": "AI in Education",
        "expected_content": DEFAULT_EXPECTED_CONTENT,
        "expected_tone": "Neutral",
        "emotion_model": "wav2vec2",
        "knowledge_source_mode": "Paste text",
        "knowledge_text": DEFAULT_KNOWLEDGE_TEXT,
        "saved_path": None,
        "transcript": "",
        "chunks": [],
        "faiss_index": None,
        "retrieved_passages": [],
        "emotion_result": None,
        "analysis_result": None,
        "attempt_saved": False,
        "latest_saved_attempt": None,
        # Sectioned recording
        "sections": [],           # list of dicts: {name, order, audio_path, transcript, local_scores}
        "sections_finalized": False,
        "active_section": "Introduction",
        "section_session_id": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def go_to_step(step_number: int) -> None:
    st.session_state.current_step = max(1, min(5, step_number))


def reset_after_reference_change() -> None:
    st.session_state.chunks = []
    st.session_state.faiss_index = None
    st.session_state.retrieved_passages = []
    st.session_state.emotion_result = None
    st.session_state.analysis_result = None
    st.session_state.attempt_saved = False
    st.session_state.latest_saved_attempt = None


def reset_all() -> None:
    for key in [
        "current_step",
        "user_id",
        "topic_name",
        "expected_content",
        "expected_tone",
        "emotion_model",
        "knowledge_source_mode",
        "knowledge_text",
        "saved_path",
        "transcript",
        "chunks",
        "faiss_index",
        "retrieved_passages",
        "emotion_result",
        "analysis_result",
        "attempt_saved",
        "latest_saved_attempt",
        "sections",
        "sections_finalized",
        "active_section",
        "section_session_id",
    ]:
        if key == "current_step":
            st.session_state[key] = 1
        elif key == "user_id":
            st.session_state[key] = "speaker_1"
        elif key == "topic_name":
            st.session_state[key] = "AI in Education"
        elif key == "expected_tone":
            st.session_state[key] = "Neutral"
        elif key == "emotion_model":
            st.session_state[key] = "wav2vec2"
        elif key == "knowledge_source_mode":
            st.session_state[key] = "Paste text"
        elif key == "expected_content":
            st.session_state[key] = DEFAULT_EXPECTED_CONTENT
        elif key == "knowledge_text":
            st.session_state[key] = DEFAULT_KNOWLEDGE_TEXT
        elif key in ["chunks", "retrieved_passages", "sections"]:
            st.session_state[key] = []
        elif key == "transcript":
            st.session_state[key] = ""
        elif key == "attempt_saved":
            st.session_state[key] = False
        elif key == "sections_finalized":
            st.session_state[key] = False
        elif key == "active_section":
            st.session_state[key] = "Introduction"
        elif key == "section_session_id":
            st.session_state[key] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            st.session_state[key] = None


# ==========================================================
# Main App
# ==========================================================

initialize_state()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center; padding: 1rem 0 0.5rem 0;">
  <div style="font-size:2.5rem;">🎙️</div>
  <div style="font-size:1.1rem; font-weight:700; color:#58a6ff; letter-spacing:0.05em;">SmartSpeak</div>
  <div style="font-size:0.72rem; color:#8b949e; margin-top:2px;">AI Speech Evaluator</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<hr style="border-color:#30363d; margin:0.75rem 0;">', unsafe_allow_html=True)

st.sidebar.markdown('<p style="font-size:0.7rem;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;">👤 Speaker Profile</p>', unsafe_allow_html=True)
sidebar_user = st.sidebar.text_input("User ID", value=st.session_state.get("user_id", "speaker_1"), label_visibility="collapsed", placeholder="User ID")
st.session_state.user_id = sidebar_user.strip() or "speaker_1"

sidebar_topic = st.sidebar.text_input("Topic", value=st.session_state.get("topic_name", "AI in Education"), label_visibility="collapsed", placeholder="Active Speech Topic")
st.session_state.topic_name = sidebar_topic.strip() or "AI in Education"

st.sidebar.markdown('<hr style="border-color:#30363d; margin:0.75rem 0;">', unsafe_allow_html=True)
st.sidebar.markdown('<p style="font-size:0.7rem;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;">🗄️ Database</p>', unsafe_allow_html=True)

conn_ok, conn_msg = test_connection()
if conn_ok:
    st.sidebar.markdown('<div style="background:rgba(63,185,80,0.12);border:1px solid rgba(63,185,80,0.4);border-radius:8px;padding:0.5rem 0.8rem;font-size:0.82rem;color:#3fb950;font-weight:600;">● MongoDB Connected</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown(f'<div style="background:rgba(240,136,62,0.1);border:1px solid rgba(240,136,62,0.35);border-radius:8px;padding:0.5rem 0.8rem;font-size:0.78rem;color:#f0883e;">⚠ {conn_msg[:60]}</div>', unsafe_allow_html=True)
    new_uri = st.sidebar.text_input("MONGODB_URI", type="password", placeholder="Paste MongoDB URI…", label_visibility="collapsed")
    if new_uri:
        cleaned_uri = re.sub(r":<([^>]+)>@", r":\1@", new_uri.strip())
        os.environ["MONGODB_URI"] = cleaned_uri
        st.sidebar.success("URI saved!")
        st.rerun()

# Step nav in sidebar
st.sidebar.markdown('<hr style="border-color:#30363d; margin:0.75rem 0;">', unsafe_allow_html=True)
st.sidebar.markdown('<p style="font-size:0.7rem;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;">📍 Navigation</p>', unsafe_allow_html=True)

STEP_ICONS = {1: "⚙️", 2: "🎙️", 3: "📝", 4: "🔍", 5: "📊"}
for sn, sl in STEPS.items():
    is_active = st.session_state.current_step == sn
    is_done = st.session_state.current_step > sn
    bg = "rgba(88,166,255,0.15)" if is_active else "rgba(63,185,80,0.08)" if is_done else "transparent"
    border = "#58a6ff" if is_active else "#3fb950" if is_done else "#30363d"
    color = "#58a6ff" if is_active else "#3fb950" if is_done else "#8b949e"
    check = " ✓" if is_done else ""
    st.sidebar.markdown(
        f'<div style="background:{bg};border:1px solid {border};border-radius:8px;padding:0.4rem 0.75rem;'
        f'margin-bottom:0.3rem;font-size:0.82rem;color:{color};font-weight:{"600" if is_active else "400"};">'
        f'{STEP_ICONS[sn]} Step {sn}: {sl}{check}</div>',
        unsafe_allow_html=True,
    )

st.sidebar.markdown('<hr style="border-color:#30363d; margin:0.75rem 0;">', unsafe_allow_html=True)

# ── Hero Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="
  background: linear-gradient(135deg, #0d1f3c 0%, #0d1117 40%, #1a1a2e 100%);
  border: 1px solid #1f6feb;
  border-radius: 16px;
  padding: 2rem 2.5rem;
  margin-bottom: 1.5rem;
  position: relative;
  overflow: hidden;
">
  <div style="position:absolute;top:-40px;right:-40px;width:200px;height:200px;
    background:radial-gradient(circle, rgba(88,166,255,0.08) 0%, transparent 70%);
    border-radius:50%;"></div>
  <div style="position:absolute;bottom:-30px;left:30%;width:150px;height:150px;
    background:radial-gradient(circle, rgba(188,140,255,0.06) 0%, transparent 70%);
    border-radius:50%;"></div>
  <div style="display:flex; align-items:center; gap:1rem; position:relative;">
    <div style="font-size:3rem; filter:drop-shadow(0 0 12px rgba(88,166,255,0.5));">🎙️</div>
    <div>
      <div style="font-size:1.9rem; font-weight:800; color:#e6edf3; letter-spacing:-0.02em; line-height:1.1;">
        SmartSpeak <span style="color:#58a6ff;">AI</span> Evaluator
      </div>
      <div style="font-size:0.88rem; color:#8b949e; margin-top:0.3rem;">
        🧠 RAG · FAISS · Whisper · Gemini &nbsp;|&nbsp; 🎯 Real-time Speech Intelligence
      </div>
    </div>
    <div style="margin-left:auto; text-align:right;">
      <div style="font-size:0.72rem; color:#484f58; text-transform:uppercase; letter-spacing:0.08em;">Current Step</div>
      <div style="font-size:1.5rem; font-weight:700; color:#58a6ff;">{step}/{total}</div>
      <div style="font-size:0.8rem; color:#8b949e;">{step_name}</div>
    </div>
  </div>
</div>
""".format(
    step=st.session_state.current_step,
    total=len(STEPS),
    step_name=STEPS[st.session_state.current_step],
), unsafe_allow_html=True)

# ── Gemini warning ───────────────────────────────────────────────────────────
if not GEMINI_API_KEY:
    st.markdown('<div style="background:rgba(240,136,62,0.1);border:1px solid rgba(240,136,62,0.35);border-radius:8px;padding:0.6rem 1rem;font-size:0.85rem;color:#f0883e;margin-bottom:1rem;">⚠️ <strong>Gemini API key not found.</strong> Local fallback scoring will be used instead of AI scoring.</div>', unsafe_allow_html=True)

# ── Step progress bar ────────────────────────────────────────────────────────
progress_value = (st.session_state.current_step - 1) / (len(STEPS) - 1)
st.progress(progress_value)
st.markdown(
    f'<div style="display:flex;justify-content:space-between;margin:-0.3rem 0 1rem 0;">'
    + "".join(
        f'<span style="font-size:0.72rem;font-weight:{"700" if i+1==st.session_state.current_step else "400"};'
        f'color:{"#58a6ff" if i+1==st.session_state.current_step else "#484f58"};">'
        f'{STEP_ICONS[i+1]} {v}</span>'
        for i, v in enumerate(STEPS.values())
    )
    + "</div>",
    unsafe_allow_html=True,
)


if st.session_state.current_step == 1:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:1.2rem;">
      <div style="width:36px;height:36px;background:linear-gradient(135deg,#1f6feb,#388bfd);border-radius:10px;
        display:flex;align-items:center;justify-content:center;font-size:1.1rem;box-shadow:0 4px 12px rgba(31,111,235,0.3);">⚙️</div>
      <div>
        <div style="font-size:1.15rem;font-weight:700;color:#e6edf3;">Reference Setup</div>
        <div style="font-size:0.78rem;color:#8b949e;">Define expected content, tone, and knowledge base for evaluation</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Personal Reference Library (MongoDB)
    with st.expander("📚 Personal Reference Library (MongoDB)", expanded=False):
        user_topics = list_reference_topics(st.session_state.user_id)
        if user_topics:
            topic_names = [t["topic_name"] for t in user_topics]
            selected_topic_name = st.selectbox("Select saved topic from library", options=topic_names, key="lib_select_topic")
            c_load, c_del = st.columns([3, 1])
            with c_load:
                if st.button("📥 Load Topic from Library", use_container_width=True):
                    found = next((t for t in user_topics if t["topic_name"] == selected_topic_name), None)
                    if found:
                        st.session_state.topic_name = found["topic_name"]
                        st.session_state.expected_content = found.get("expected_content", "")
                        st.session_state.expected_tone = found.get("expected_tone", "Neutral")
                        st.session_state.knowledge_text = found.get("knowledge_text", "")
                        reset_after_reference_change()
                        st.success(f"Loaded topic '{found['topic_name']}'!")
                        st.rerun()
            with c_del:
                if st.button("🗑️ Delete Topic", use_container_width=True):
                    delete_reference_topic(st.session_state.user_id, selected_topic_name)
                    st.warning(f"Deleted '{selected_topic_name}' from library.")
                    st.rerun()
        else:
            st.info("No saved topics in your library yet. Fill in the fields below and click 'Save to Library'!")

        st.markdown("---")
        lib_c1, lib_c2 = st.columns([3, 1])
        with lib_c1:
            save_as_name = st.text_input("Save current topic as", value=st.session_state.get("topic_name", "AI in Education"), key="lib_save_name")
        with lib_c2:
            st.write("")
            st.write("")
            if st.button("💾 Save to Library", use_container_width=True):
                if not st.session_state.expected_content.strip() or not st.session_state.knowledge_text.strip():
                    st.error("Please fill Expected Content and Knowledge Text before saving.")
                else:
                    ok = save_reference_topic(
                        user_id=st.session_state.user_id,
                        topic_name=save_as_name,
                        expected_content=st.session_state.expected_content.strip(),
                        expected_tone=st.session_state.expected_tone,
                        knowledge_text=st.session_state.knowledge_text.strip(),
                    )
                    if ok:
                        st.session_state.topic_name = save_as_name
                        st.success(f"Topic '{save_as_name}' saved to your MongoDB library!")
                        st.rerun()
                    else:
                        st.error("Unable to save to MongoDB. Please check MONGODB_URI.")

    expected_input = st.text_area(
        "Expected speech content",
        value=st.session_state.expected_content,
        height=130,
        placeholder="Describe what the speaker is expected to say.",
    )

    tone_options = ["Any", "Neutral", "Happy", "Sad", "Angry"]
    tone_index = tone_options.index(st.session_state.expected_tone) if st.session_state.expected_tone in tone_options else 0
    expected_tone_input = st.selectbox(
        "Expected tone (for tone scoring)",
        options=tone_options,
        index=tone_index,
    )

    model_labels = [
        "Gemini-3.5-flash-lite",
        "Wav2Vec 2.0 (Local transformer)",
        "CNN",
        "LSTM",
    ]
    model_values = ["gemini", "wav2vec2", "cnn", "lstm"]
    saved_model = st.session_state.get("emotion_model", "gemini")
    model_idx = model_values.index(saved_model) if saved_model in model_values else 0
    selected_model_label = st.selectbox(
        "Tone / Emotion recognition model",
        options=model_labels,
        index=model_idx,
        help=(
            "Gemini: sends all section audio in ONE API call — fast and quota-efficient. "
            "Wav2Vec: runs locally but loads a large transformer per section (slow for multi-section). "
            "CNN/LSTM: legacy models via Flask backend."
        ),
    )
    selected_model_value = model_values[model_labels.index(selected_model_label)]

    source_mode = st.radio(
        "Knowledge base source for RAG",
        options=["Paste text", "Upload file (.txt, .md, .csv)"],
        index=0 if st.session_state.knowledge_source_mode == "Paste text" else 1,
        horizontal=True,
    )

    kb_text = ""
    if source_mode == "Paste text":
        kb_text = st.text_area(
            "Paste knowledge text",
            value=st.session_state.knowledge_text if st.session_state.knowledge_source_mode == "Paste text" else "",
            height=220,
            placeholder="Paste source content used for retrieval and evidence-based scoring.",
        )
    else:
        uploaded_doc = st.file_uploader("Upload text-based file", type=["txt", "md", "csv"])
        if uploaded_doc is not None:
            kb_text = read_text_file(uploaded_doc)
            st.text_area("Extracted file preview", value=kb_text[:3000], height=220, disabled=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Save Step 1", type="primary", use_container_width=True):
            if not expected_input.strip():
                st.error("Expected speech content is required.")
            elif not kb_text.strip():
                st.error("Knowledge base text is required for RAG.")
            else:
                st.session_state.expected_content = expected_input.strip()
                st.session_state.expected_tone = expected_tone_input
                st.session_state.emotion_model = selected_model_value
                st.session_state.knowledge_text = kb_text.strip()
                st.session_state.knowledge_source_mode = "Paste text" if source_mode == "Paste text" else "Upload file"
                reset_after_reference_change()
                st.success("Step 1 saved successfully.")

    with col_b:
        if st.button("Next Step -> Audio Input", use_container_width=True):
            if not st.session_state.expected_content or not st.session_state.knowledge_text:
                st.warning("Please save Step 1 first.")
            else:
                go_to_step(2)
                st.rerun()


elif st.session_state.current_step == 2:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:1.2rem;">
      <div style="width:36px;height:36px;background:linear-gradient(135deg,#8b5cf6,#a78bfa);border-radius:10px;
        display:flex;align-items:center;justify-content:center;font-size:1.1rem;box-shadow:0 4px 12px rgba(139,92,246,0.3);">🎙️</div>
      <div>
        <div style="font-size:1.15rem;font-weight:700;color:#e6edf3;">Record Speech in Sections</div>
        <div style="font-size:0.78rem;color:#8b949e;">Record each section separately — Introduction, Body, Conclusion</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption(
        "Record each section (Introduction, Body, Conclusion) separately. "
        "You can re-record any section. Finalize when all sections are ready."
    )

    PREDEFINED_SECTIONS = ["Introduction", "Body", "Conclusion"]

    # --- Visual Section Status Tracker ---
    sec_names_recorded = {s["name"] for s in st.session_state.sections}
    status_pills_html = '<div style="display:flex;gap:0.75rem;margin-bottom:1.5rem;flex-wrap:wrap;">'
    for ps in PREDEFINED_SECTIONS:
        is_rec = ps in sec_names_recorded
        dot = "🟢" if is_rec else "⚪"
        badge_bg = "rgba(63,185,80,0.12)" if is_rec else "rgba(48,54,61,0.4)"
        badge_border = "#3fb950" if is_rec else "#30363d"
        badge_color = "#3fb950" if is_rec else "#8b949e"
        status_text = "Recorded" if is_rec else "Pending"
        status_pills_html += f'''
        <div style="background:{badge_bg};border:1px solid {badge_border};border-radius:10px;padding:0.45rem 1rem;display:flex;align-items:center;gap:0.5rem;">
          <span>{dot}</span>
          <span style="font-weight:600;color:#e6edf3;font-size:0.85rem;">{ps}</span>
          <span style="font-size:0.72rem;color:{badge_color};text-transform:uppercase;font-weight:600;letter-spacing:0.04em;">{status_text}</span>
        </div>
        '''
    status_pills_html += '</div>'
    st.markdown(status_pills_html, unsafe_allow_html=True)

    # --- Section selector ---
    col_sel, col_custom = st.columns([2, 2])
    with col_sel:
        active_choice = st.selectbox(
            "Select Section to Record / Update",
            options=PREDEFINED_SECTIONS + ["Custom…"],
            index=PREDEFINED_SECTIONS.index(st.session_state.active_section)
            if st.session_state.active_section in PREDEFINED_SECTIONS
            else len(PREDEFINED_SECTIONS),
            key="section_choice",
        )
    with col_custom:
        custom_name = ""
        if active_choice == "Custom…":
            custom_name = st.text_input("Custom Section Name", key="custom_section_name", placeholder="e.g. Q&A, Call to Action")
        else:
            st.write("")

    section_name = custom_name.strip() if active_choice == "Custom…" else active_choice
    if section_name:
        st.session_state.active_section = section_name

    # --- Per-section audio input card ---
    if section_name:
        st.markdown(f"""
        <div style="background:linear-gradient(145deg,#1c2333 0%,#161b22 100%);border:1px solid #30363d;border-radius:12px;padding:1.2rem;margin-bottom:1rem;">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.75rem;">
            <div style="font-size:1.05rem;font-weight:700;color:#e6edf3;">🎙️ Recording Studio: <span style="color:#58a6ff;">{section_name}</span></div>
            <div style="font-size:0.75rem;background:rgba(88,166,255,0.12);color:#58a6ff;border:1px solid rgba(88,166,255,0.3);border-radius:20px;padding:2px 10px;font-weight:600;">ACTIVE SLICE</div>
          </div>
        """, unsafe_allow_html=True)

        existing = next(
            (s for s in st.session_state.sections if s["name"] == section_name), None
        )
        if existing:
            st.info(f"✅ **{section_name}** already has a recording. Record again below to replace it.")
            st.audio(load_audio_bytes(Path(existing["audio_path"])))

        audio_input = st.audio_input(f"Record audio for {section_name}", key=f"audio_{section_name}")
        uploaded_audio = st.file_uploader(
            f"Or upload audio file for {section_name}",
            type=["wav", "mp3", "m4a", "flac"],
            key=f"upload_{section_name}",
        )

        raw_bytes = None
        suffix = ".wav"
        if audio_input is not None:
            raw_bytes = audio_input.getvalue()
        elif uploaded_audio is not None:
            raw_bytes = uploaded_audio.getvalue()
            suffix = Path(uploaded_audio.name).suffix or ".wav"

        if st.button(f"💾 Save & Lock '{section_name}' Section", type="primary", use_container_width=True):
            if not section_name:
                st.error("Please enter a section name.")
            elif raw_bytes is None:
                st.error("Please record or upload audio for this section first.")
            else:
                try:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_name = re.sub(r"[^A-Za-z0-9_]", "_", section_name)
                    file_path = RECORDINGS_DIR / f"section_{safe_name}_{timestamp}{suffix}"
                    with open(file_path, "wb") as fh:
                        fh.write(raw_bytes)

                    if section_name in PREDEFINED_SECTIONS:
                        order = PREDEFINED_SECTIONS.index(section_name)
                    else:
                        existing_orders = [s["order"] for s in st.session_state.sections]
                        order = max(existing_orders, default=len(PREDEFINED_SECTIONS) - 1) + 1

                    idx = next(
                        (i for i, s in enumerate(st.session_state.sections) if s["name"] == section_name),
                        None,
                    )
                    section_entry = {
                        "name": section_name,
                        "order": order,
                        "audio_path": str(file_path),
                        "transcript": "",
                        "local_scores": None,
                    }
                    if idx is not None:
                        st.session_state.sections[idx] = section_entry
                    else:
                        st.session_state.sections.append(section_entry)

                    st.session_state.sections.sort(key=lambda s: s["order"])

                    save_speech_section(
                        user_id=st.session_state.user_id,
                        topic_name=st.session_state.topic_name,
                        session_id=st.session_state.section_session_id,
                        section_name=section_name,
                        audio_path=str(file_path),
                        transcript="",
                        order=order,
                    )

                    st.session_state.sections_finalized = False
                    st.session_state.transcript = ""
                    st.success(f"Section '{section_name}' saved successfully! ✅")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to save section: {exc}")

        st.markdown("</div>", unsafe_allow_html=True)

    # --- Sections overview cards ---
    if st.session_state.sections:
        st.markdown("""
        <div style="font-size:1rem;font-weight:700;color:#e6edf3;margin:1.5rem 0 0.8rem 0;display:flex;align-items:center;gap:0.5rem;">
          <span>📋 Recorded Sections Track</span>
          <span style="font-size:0.75rem;background:rgba(63,185,80,0.15);color:#3fb950;border-radius:12px;padding:2px 8px;">{count} Ready</span>
        </div>
        """.format(count=len(st.session_state.sections)), unsafe_allow_html=True)

        for i, sec in enumerate(st.session_state.sections):
            card_col1, card_col2, card_col3 = st.columns([3, 4, 1])
            with card_col1:
                st.markdown(f"""
                <div style="background:var(--bg-card);border:1px solid #30363d;border-radius:10px;padding:0.75rem 1rem;">
                  <div style="font-weight:700;color:#58a6ff;font-size:0.95rem;">#{i+1} {sec['name']}</div>
                  <div style="font-size:0.72rem;color:#3fb950;">● Audio Locked</div>
                </div>
                """, unsafe_allow_html=True)
            with card_col2:
                try:
                    st.audio(load_audio_bytes(Path(sec["audio_path"])))
                except Exception:
                    st.caption("File missing")
            with card_col3:
                if st.button("🗑️ Delete", key=f"del_{i}_{sec['name']}", use_container_width=True):
                    st.session_state.sections.pop(i)
                    st.session_state.sections_finalized = False
                    st.session_state.transcript = ""
                    st.rerun()

        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

        # --- Finalize CTA Banner ---
        if not st.session_state.sections_finalized:
            st.markdown("""
            <div style="background:linear-gradient(135deg,rgba(31,111,235,0.15) 0%,rgba(188,140,255,0.1) 100%);border:1px solid #1f6feb;border-radius:12px;padding:1.2rem;margin-top:1rem;text-align:center;">
              <div style="font-size:1.05rem;font-weight:700;color:#e6edf3;margin-bottom:0.25rem;">Ready to Transcribe?</div>
              <div style="font-size:0.8rem;color:#8b949e;margin-bottom:0.75rem;">Lock all speech sections to merge and run high-accuracy Whisper speech-to-text.</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🚀 Finalize All Sections & Proceed to Transcription", type="primary", use_container_width=True):
                st.session_state.sections_finalized = True
                if st.session_state.sections:
                    st.session_state.saved_path = Path(st.session_state.sections[0]["audio_path"])
                st.session_state.transcript = ""
                st.success("All sections finalized! Proceed to Transcription.")
                st.rerun()
        else:
            st.markdown('<div style="background:rgba(63,185,80,0.12);border:1px solid rgba(63,185,80,0.4);border-radius:10px;padding:0.75rem 1.2rem;color:#3fb950;font-weight:600;font-size:0.9rem;margin-top:0.75rem;">✅ All sections finalized & locked. You are ready to transcribe!</div>', unsafe_allow_html=True)

        with st.expander("🔄 Or switch to a single recording take (Overrides sections)", expanded=False):
            single_audio = st.file_uploader("Upload full speech audio", type=["wav", "mp3", "m4a", "flac"], key="single_full_audio")
            if single_audio is not None:
                suffix = Path(single_audio.name).suffix or ".wav"
                try:
                    st.session_state.saved_path = save_audio_bytes(single_audio.getvalue(), suffix=suffix)
                    st.session_state.sections = []
                    st.session_state.sections_finalized = False
                    st.success(f"Single audio saved: {st.session_state.saved_path.name}")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
    else:
        st.markdown("""
        <div style="background:var(--bg-card);border:1px dashed #30363d;border-radius:12px;padding:1.5rem;text-align:center;margin-top:1rem;">
          <div style="font-size:1.8rem;margin-bottom:0.5rem;">🎙️</div>
          <div style="font-weight:600;color:#e6edf3;">No sections recorded yet</div>
          <div style="font-size:0.8rem;color:#8b949e;margin-top:0.25rem;">Record your Introduction, Body, or Conclusion above to begin building your sectioned speech.</div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("🔄 Single-take recording mode", expanded=False):
            single_mic = st.audio_input("Record your full speech (single take)", key="single_mic_step2")
            single_up = st.file_uploader("Or upload audio file", type=["wav", "mp3", "m4a", "flac"], key="single_up_step2")
            raw_single = None
            suffix_single = ".wav"
            if single_mic is not None:
                raw_single = single_mic.getvalue()
            elif single_up is not None:
                raw_single = single_up.getvalue()
                suffix_single = Path(single_up.name).suffix or ".wav"
            if raw_single is not None:
                try:
                    st.session_state.saved_path = save_audio_bytes(raw_single, suffix=suffix_single)
                    st.session_state.sections_finalized = True
                    st.success(f"Audio saved: {st.session_state.saved_path.name}")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    if st.session_state.saved_path and not st.session_state.sections:
        st.audio(load_audio_bytes(st.session_state.saved_path))
        st.caption(f"Current audio file: {Path(st.session_state.saved_path).name}")

    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("<- Back to Step 1", use_container_width=True):
            go_to_step(1)
            st.rerun()
    with col_next:
        can_proceed = st.session_state.sections_finalized and (
            bool(st.session_state.sections) or bool(st.session_state.saved_path)
        )
        if st.button("Next Step -> Transcription", type="primary", use_container_width=True):
            if not can_proceed:
                st.warning("Please record at least one section (or a single audio) and finalize before proceeding.")
            else:
                go_to_step(3)
                st.rerun()


elif st.session_state.current_step == 3:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:1.2rem;">
      <div style="width:36px;height:36px;background:linear-gradient(135deg,#0ea5e9,#38bdf8);border-radius:10px;
        display:flex;align-items:center;justify-content:center;font-size:1.1rem;box-shadow:0 4px 12px rgba(14,165,233,0.3);">📝</div>
      <div>
        <div style="font-size:1.15rem;font-weight:700;color:#e6edf3;">Transcribe Speech</div>
        <div style="font-size:0.78rem;color:#8b949e;">Whisper AI converts audio to text — section by section</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    has_sections = bool(st.session_state.sections)

    if has_sections:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(14,165,233,0.12) 0%,rgba(56,189,248,0.06) 100%);border:1px solid rgba(14,165,233,0.3);border-radius:12px;padding:1rem 1.25rem;margin-bottom:1.25rem;display:flex;align-items:center;justify-content:space-between;">
          <div style="display:flex;align-items:center;gap:0.75rem;">
            <div style="font-size:1.5rem;">🗂️</div>
            <div>
              <div style="font-weight:700;color:#e6edf3;font-size:0.95rem;">Multi-Section Pipeline Ready</div>
              <div style="font-size:0.78rem;color:#8b949e;">{len(st.session_state.sections)} section(s) loaded: <strong>{", ".join(s["name"] for s in st.session_state.sections)}</strong></div>
            </div>
          </div>
          <div style="font-size:0.75rem;background:rgba(14,165,233,0.2);color:#38bdf8;border:1px solid #0ea5e9;border-radius:20px;padding:4px 12px;font-weight:700;">WHISPER ASR</div>
        </div>
        """, unsafe_allow_html=True)

        # Show per-section audio preview in 2-column bento
        preview_cols = st.columns(min(3, len(st.session_state.sections)))
        for idx, sec in enumerate(st.session_state.sections):
            col_target = preview_cols[idx % len(preview_cols)]
            with col_target:
                st.markdown(f"""
                <div style="background:var(--bg-card);border:1px solid #30363d;border-radius:10px;padding:0.75rem;margin-bottom:0.5rem;">
                  <div style="font-size:0.85rem;font-weight:700;color:#38bdf8;margin-bottom:0.25rem;">#{idx+1} {sec['name']}</div>
                </div>
                """, unsafe_allow_html=True)
                try:
                    st.audio(load_audio_bytes(Path(sec["audio_path"])))
                except Exception:
                    st.caption("Audio missing")

        if st.button("✨ Transcribe All Sections via Whisper", type="primary", use_container_width=True):
            combined_parts = []
            with st.status("Transcribing sections with Whisper...", expanded=True) as status:
                try:
                    for i, sec in enumerate(st.session_state.sections):
                        status.write(f"🎙️ Processing section {i+1}/{len(st.session_state.sections)}: **{sec['name']}**…")
                        sec_text = speech_to_text(Path(sec["audio_path"]))
                        st.session_state.sections[i]["transcript"] = sec_text
                        combined_parts.append(f"[{sec['name']}] {sec_text}")

                        # Update MongoDB section with transcript (best-effort)
                        save_speech_section(
                            user_id=st.session_state.user_id,
                            topic_name=st.session_state.topic_name,
                            session_id=st.session_state.section_session_id,
                            section_name=sec["name"],
                            audio_path=sec["audio_path"],
                            transcript=sec_text,
                            order=sec["order"],
                        )

                    st.session_state.transcript = "\n\n".join(combined_parts)
                    st.session_state.retrieved_passages = []
                    st.session_state.emotion_result = None
                    st.session_state.analysis_result = None
                    status.update(label="All sections transcribed successfully! ✅", state="complete")
                except Exception as exc:
                    status.update(label="Transcription failed.", state="error")
                    st.error(str(exc))

        # Show per-section transcripts with badges
        if any(s.get("transcript") for s in st.session_state.sections):
            st.markdown('<div style="font-weight:700;color:#e6edf3;font-size:1.05rem;margin:1.5rem 0 0.8rem 0;">📄 Individual Section Transcripts</div>', unsafe_allow_html=True)
            for sec in st.session_state.sections:
                if sec.get("transcript"):
                    word_cnt = len(sec["transcript"].split())
                    st.markdown(f"""
                    <div style="background:var(--bg-card);border:1px solid #30363d;border-left:3px solid #38bdf8;border-radius:10px;padding:0.9rem 1.1rem;margin-bottom:0.75rem;">
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">
                        <span style="font-weight:700;color:#58a6ff;font-size:0.9rem;">{sec['name']}</span>
                        <span style="font-size:0.72rem;color:#8b949e;background:#161b22;padding:2px 8px;border-radius:6px;border:1px solid #30363d;">{word_cnt} words</span>
                      </div>
                      <div style="color:#e6edf3;font-size:0.875rem;line-height:1.6;font-family:'Inter',sans-serif;">{sec['transcript']}</div>
                    </div>
                    """, unsafe_allow_html=True)

        if st.session_state.transcript:
            total_words = len(st.session_state.transcript.split())
            st.markdown(f"""
            <div style="background:linear-gradient(145deg,#1c2333 0%,#21293a 100%);border:1px solid #58a6ff;border-radius:12px;padding:1.2rem;margin-top:1.2rem;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;">
                <div style="font-weight:700;color:#e6edf3;font-size:1rem;display:flex;align-items:center;gap:0.4rem;">
                  <span>📝 Full Concatenated Speech Transcript</span>
                </div>
                <div style="font-size:0.75rem;background:rgba(88,166,255,0.15);color:#58a6ff;border:1px solid rgba(88,166,255,0.3);border-radius:12px;padding:2px 10px;font-weight:700;">
                  {total_words} TOTAL WORDS
                </div>
              </div>
              <div style="color:#e6edf3;font-size:0.9rem;line-height:1.7;white-space:pre-wrap;background:#0d1117;padding:1rem;border-radius:8px;border:1px solid #30363d;max-height:220px;overflow-y:auto;">{st.session_state.transcript}</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        # --- Classic single-audio path (backward compatible) ---
        if not st.session_state.saved_path:
            st.warning("No audio found. Please return to Step 2 to record or upload.")
        else:
            st.audio(load_audio_bytes(st.session_state.saved_path))

            if st.button("▶️ Transcribe Full Speech via Whisper", type="primary", use_container_width=True):
                with st.status("Transcribing with Whisper...", expanded=True) as status:
                    try:
                        transcript = speech_to_text(st.session_state.saved_path)
                        st.session_state.transcript = transcript
                        st.session_state.retrieved_passages = []
                        st.session_state.emotion_result = None
                        st.session_state.analysis_result = None
                        status.update(label="Transcription completed! ✅", state="complete")
                    except Exception as exc:
                        status.update(label="Transcription failed.", state="error")
                        st.error(str(exc))

            if st.session_state.transcript:
                st.markdown(f"""
                <div style="background:var(--bg-card);border:1px solid #58a6ff;border-radius:12px;padding:1.1rem;margin-top:1rem;">
                  <div style="font-weight:700;color:#58a6ff;margin-bottom:0.5rem;">Transcript</div>
                  <div style="color:#e6edf3;font-size:0.9rem;line-height:1.6;white-space:pre-wrap;">{st.session_state.transcript}</div>
                </div>
                """, unsafe_allow_html=True)

    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("<- Back to Step 2", use_container_width=True):
            go_to_step(2)
            st.rerun()
    with col_next:
        if st.button("Next Step -> Build RAG", type="primary", use_container_width=True):
            if not st.session_state.transcript.strip():
                st.warning("Please transcribe audio first.")
            else:
                go_to_step(4)
                st.rerun()


elif st.session_state.current_step == 4:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:1rem;">
      <div style="width:36px;height:36px;background:linear-gradient(135deg,#f59e0b,#fbbf24);border-radius:10px;
        display:flex;align-items:center;justify-content:center;font-size:1.1rem;box-shadow:0 4px 12px rgba(245,158,11,0.3);">🔍</div>
      <div>
        <div style="font-size:1.15rem;font-weight:700;color:#e6edf3;">Build FAISS Index & Retrieve Context</div>
        <div style="font-size:0.78rem;color:#8b949e;">Semantic search over your knowledge base using vector embeddings</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("RAG retrieval uses sentence-transformers embeddings and FAISS similarity search.")

    # Vector Pipeline Overview Stats
    kb_len = len(st.session_state.knowledge_text)
    trans_len = len(st.session_state.transcript)
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:0.75rem;margin:1rem 0 1.25rem 0;">
      <div style="background:var(--bg-card);border:1px solid #30363d;border-radius:10px;padding:0.85rem 1rem;">
        <div style="font-size:0.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:0.05em;font-weight:600;">📚 Knowledge Corpus</div>
        <div style="font-size:1.25rem;font-weight:700;color:#e6edf3;margin-top:0.2rem;">{kb_len:,} <span style="font-size:0.75rem;color:#8b949e;font-weight:400;">chars</span></div>
      </div>
      <div style="background:var(--bg-card);border:1px solid #30363d;border-radius:10px;padding:0.85rem 1rem;">
        <div style="font-size:0.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:0.05em;font-weight:600;">📝 Query Transcript</div>
        <div style="font-size:1.25rem;font-weight:700;color:#e6edf3;margin-top:0.2rem;">{trans_len:,} <span style="font-size:0.75rem;color:#8b949e;font-weight:400;">chars</span></div>
      </div>
      <div style="background:var(--bg-card);border:1px solid #30363d;border-radius:10px;padding:0.85rem 1rem;">
        <div style="font-size:0.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:0.05em;font-weight:600;">⚡ Embedding Model</div>
        <div style="font-size:0.95rem;font-weight:700;color:#58a6ff;margin-top:0.3rem;">all-MiniLM-L6-v2</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Index Knowledge Base & Retrieve Semantic Evidence", type="primary", use_container_width=True):
        with st.status("Building FAISS vector index and executing similarity search...", expanded=True) as status:
            try:
                st.session_state.chunks = chunk_text(st.session_state.knowledge_text)
                if not st.session_state.chunks:
                    raise ValueError("No chunks were generated from knowledge text.")

                index, _ = build_faiss_index(st.session_state.chunks)
                st.session_state.faiss_index = index

                query = f"Expected: {st.session_state.expected_content}\nSpoken: {st.session_state.transcript}"
                st.session_state.retrieved_passages = retrieve_context(
                    query=query,
                    index=st.session_state.faiss_index,
                    chunks=st.session_state.chunks,
                    k=4,
                )
                status.update(label="Semantic evidence retrieved successfully! ✅", state="complete")
            except Exception as exc:
                status.update(label="RAG retrieval failed.", state="error")
                st.error(str(exc))

    if st.session_state.retrieved_passages:
        st.markdown("""
        <div style="font-size:1rem;font-weight:700;color:#e6edf3;margin:1.5rem 0 0.8rem 0;display:flex;align-items:center;gap:0.5rem;">
          <span>🎯 Top Semantic Evidence Passages</span>
          <span style="font-size:0.75rem;background:rgba(57,208,216,0.15);color:#39d0d8;border-radius:12px;padding:2px 8px;font-weight:700;">FAISS Top-4</span>
        </div>
        """, unsafe_allow_html=True)

        for i, passage in enumerate(st.session_state.retrieved_passages, start=1):
            sim_pct = round(passage['similarity'] * 100, 1)
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid #30363d;border-left:3px solid #39d0d8;border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.75rem;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">
                <span style="font-weight:700;color:#39d0d8;font-size:0.85rem;">Rank #{i} • Chunk {passage['chunk_id']}</span>
                <span style="font-size:0.75rem;font-weight:700;background:rgba(57,208,216,0.15);color:#39d0d8;border:1px solid rgba(57,208,216,0.3);border-radius:20px;padding:2px 10px;">
                  {sim_pct}% SIMILARITY
                </span>
              </div>
              <div style="color:#e6edf3;font-size:0.875rem;line-height:1.6;font-style:italic;">"{passage['text']}"</div>
            </div>
            """, unsafe_allow_html=True)

    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("<- Back to Step 3", use_container_width=True):
            go_to_step(3)
            st.rerun()
    with col_next:
        if st.button("Next Step -> Detailed Scoring", type="primary", use_container_width=True):
            if not st.session_state.retrieved_passages:
                st.warning("Run RAG retrieval first.")
            else:
                go_to_step(5)
                st.rerun()


elif st.session_state.current_step == 5:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:1.2rem;">
      <div style="width:36px;height:36px;background:linear-gradient(135deg,#10b981,#34d399);border-radius:10px;
        display:flex;align-items:center;justify-content:center;font-size:1.1rem;box-shadow:0 4px 12px rgba(16,185,129,0.3);">📊</div>
      <div>
        <div style="font-size:1.15rem;font-weight:700;color:#e6edf3;">Detailed Analysis & Scoring</div>
        <div style="font-size:0.78rem;color:#8b949e;">AI-powered evaluation with RAG evidence and tone recognition</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Run Detailed Analysis", type="primary"):
        with st.status("Scoring transcript with RAG evidence...", expanded=True) as status:
            try:
                st.session_state.analysis_result = llm_scoring(
                    expected=st.session_state.expected_content,
                    transcript=st.session_state.transcript,
                    retrieved=st.session_state.retrieved_passages,
                )

                selected_model = st.session_state.get("emotion_model", "gemini")

                # --- Tone prediction ---
                # Gemini path: ONE API call for ALL sections (fast, quota-efficient)
                if selected_model == "gemini" and GEMINI_API_KEY:
                    if st.session_state.sections:
                        # Multi-section: send all audio files in a single Gemini call
                        status.write(f"🎙️ Predicting tone via Gemini for {len(st.session_state.sections)} section(s) in 1 API call…")
                        audio_paths = [Path(s["audio_path"]) for s in st.session_state.sections]
                    else:
                        # Single audio fallback
                        status.write("🎙️ Predicting tone via Gemini…")
                        audio_paths = [Path(st.session_state.saved_path)]

                    try:
                        st.session_state.emotion_result = predict_tone_gemini(audio_paths)
                    except Exception as gemini_exc:
                        status.write(f"⚠️ Gemini tone failed ({gemini_exc}), falling back to Wav2Vec…")
                        wav_path = get_wav_for_emotion(Path(st.session_state.saved_path))
                        st.session_state.emotion_result = predict_emotion_wav2vec2_local(wav_path)

                else:
                    # Wav2Vec / CNN / LSTM path (single-file, legacy)
                    wav_path = get_wav_for_emotion(Path(st.session_state.saved_path))
                    st.session_state.emotion_result = call_emotion_service(wav_path, model_name=selected_model)

                # Per-section local scoring (no extra Gemini calls — saves weekly limit)
                if st.session_state.sections:
                    status.write("Computing per-section scores (local)…")
                    for i, sec in enumerate(st.session_state.sections):
                        sec_transcript = sec.get("transcript", "")
                        if sec_transcript.strip():
                            sec_scores = local_scoring(
                                st.session_state.expected_content,
                                sec_transcript,
                                st.session_state.retrieved_passages,
                            )
                            st.session_state.sections[i]["local_scores"] = sec_scores

                status.update(label="Detailed scoring completed.", state="complete")
            except Exception as exc:
                status.update(label="One or more scoring components failed. Running robust fallback.", state="error")
                st.warning(str(exc))
                if st.session_state.analysis_result is None:
                    st.session_state.analysis_result = local_scoring(
                        st.session_state.expected_content,
                        st.session_state.transcript,
                        st.session_state.retrieved_passages,
                    )
                if st.session_state.emotion_result is None:
                    st.session_state.emotion_result = {
                        "emotion_label": "Unknown",
                        "confidence": 0.0,
                        "probabilities": {},
                    }

    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        overall = int(result.get("overall_score", 0))
        scores = result.get("scores", {})
        emotion = st.session_state.emotion_result or {}
        predicted_tone = str(emotion.get("emotion_label", "Unknown"))
        tone_confidence = float(emotion.get("confidence", 0.0))
        model_used = emotion.get("model", st.session_state.get("emotion_model", "wav2vec2"))
        tone_score = tone_score_from_result(
            predicted_tone=predicted_tone,
            confidence=tone_confidence,
            expected_tone=st.session_state.expected_tone,
        )
        combined_score = merge_context_and_tone(overall, tone_score, tone_weight=0.25)

        # Auto-save practice attempt to MongoDB (includes sections)
        if not st.session_state.get("attempt_saved", False):
            try:
                full_scores = {
                    "overall_score": overall,
                    "topic_relevance": int(scores.get("topic_relevance", 0)),
                    "key_point_coverage": int(scores.get("key_point_coverage", 0)),
                    "content_accuracy": int(scores.get("content_accuracy", 0)),
                    "clarity": int(scores.get("clarity", 0)),
                    "combined_score": combined_score,
                }
                tone_info = {
                    "predicted_tone": predicted_tone,
                    "confidence": tone_confidence,
                    "tone_score": tone_score,
                    "probabilities": emotion.get("probabilities", {}),
                    "model": model_used,
                }
                # Build sections summary for MongoDB
                sections_for_db = [
                    {
                        "name": s["name"],
                        "order": s.get("order", 0),
                        "transcript": s.get("transcript", ""),
                        "scores": s.get("local_scores", {}).get("scores", {}) if s.get("local_scores") else {},
                        "overall_score": s.get("local_scores", {}).get("overall_score", 0) if s.get("local_scores") else 0,
                    }
                    for s in st.session_state.sections
                ]
                saved_doc = save_practice_attempt(
                    user_id=st.session_state.user_id,
                    topic_name=st.session_state.get("topic_name", "AI in Education"),
                    expected_content=st.session_state.expected_content,
                    transcript=st.session_state.transcript,
                    scores=full_scores,
                    tone=tone_info,
                    step_analysis=result.get("step_by_step_analysis", []),
                    strengths=result.get("strengths", []),
                    improvements=result.get("improvements", []),
                    final_summary=result.get("final_summary", ""),
                    sections=sections_for_db if sections_for_db else None,
                )
                if saved_doc:
                    st.session_state.latest_saved_attempt = saved_doc
                    st.session_state.attempt_saved = True
            except Exception:
                pass

        # Progress Delta / Comparison Card
        latest_att = st.session_state.get("latest_saved_attempt")
        if latest_att:
            att_number = latest_att.get("attempt_number", 1)
            delta = latest_att.get("delta_vs_previous", {})

            st.markdown(f"### 📈 Practice Progress (Attempt #{att_number})")

            if delta and delta.get("prev_attempt_number"):
                prev_n = delta["prev_attempt_number"]
                ov_diff = delta.get("overall", 0)
                clarity_diff = delta.get("clarity", 0)
                cov_diff = delta.get("key_point_coverage", 0)
                rel_diff = delta.get("topic_relevance", 0)

                fmt_diff = lambda d: f"+{d}" if d > 0 else f"{d}"
                c_d1, c_d2, c_d3, c_d4 = st.columns(4)
                c_d1.metric("Overall Change", f"{fmt_diff(ov_diff)} pts", delta=ov_diff)
                c_d2.metric("Clarity Delta", f"{fmt_diff(clarity_diff)}%", delta=clarity_diff)
                c_d3.metric("Coverage Delta", f"{fmt_diff(cov_diff)}%", delta=cov_diff)
                c_d4.metric("Relevance Delta", f"{fmt_diff(rel_diff)}%", delta=rel_diff)

                if ov_diff > 0:
                    st.success(f"🎉 **Great job!** You improved your overall score by **+{ov_diff} points** compared to Attempt #{prev_n}!")
                elif ov_diff < 0:
                    st.warning(f"⚠️ Your score decreased by **{ov_diff} points** compared to Attempt #{prev_n}. Check the improvements section below.")
                else:
                    st.info(f"Consistent score with Attempt #{prev_n}.")
            else:
                st.info(
                    f"✨ Recorded **Attempt #1** for topic **'{st.session_state.get('topic_name')}'**. "
                    "When you record your next attempt, your progress and improvement deltas will appear here!"
                )

            # History table
            all_attempts = list_practice_attempts(st.session_state.user_id, st.session_state.get("topic_name"))
            if len(all_attempts) > 1:
                with st.expander(f"📊 View All Attempts History ({len(all_attempts)} attempts recorded)", expanded=False):
                    rows = []
                    for h in all_attempts:
                        h_s = h.get("scores", {})
                        h_tone = h.get("tone", {})
                        rows.append({
                            "Attempt": f"#{h.get('attempt_number', 1)}",
                            "Overall Score": f"{int(h_s.get('overall_score', 0))}/100",
                            "Clarity": f"{int(h_s.get('clarity', 0))}/100",
                            "Coverage": f"{int(h_s.get('key_point_coverage', 0))}/100",
                            "Relevance": f"{int(h_s.get('topic_relevance', 0))}/100",
                            "Tone": h_tone.get("predicted_tone", "N/A"),
                            "Recorded At": str(h.get("created_at", ""))[:19],
                        })
                    st.dataframe(rows, use_container_width=True)
            st.markdown("---")

        # ── Executive Scorecard Hero Bento ────────────────────────────────
        tier_label = "🏆 World-Class Delivery" if overall >= 85 else "🌟 Proficient Speaker" if overall >= 70 else "📈 Developing Impact" if overall >= 50 else "🌱 Foundational Stage"
        tier_color = "#3fb950" if overall >= 85 else "#58a6ff" if overall >= 70 else "#f0883e" if overall >= 50 else "#f85149"
        tier_bg = "rgba(63,185,80,0.12)" if overall >= 85 else "rgba(88,166,255,0.12)" if overall >= 70 else "rgba(240,136,62,0.12)" if overall >= 50 else "rgba(248,81,73,0.12)"

        tone_emojis = {"Neutral": "😐", "Happy": "😊", "Angry": "😠", "Sad": "😢"}
        tone_emoji = tone_emojis.get(predicted_tone, "🎙️")
        tone_aligned = (predicted_tone.lower() == st.session_state.expected_tone.lower()) or (st.session_state.expected_tone.lower() == "any")
        align_badge = "✅ Aligned" if tone_aligned else "⚠️ Tone Variance"
        align_color = "#3fb950" if tone_aligned else "#f0883e"

        score_col1, score_col2 = st.columns([1, 1])
        with score_col1:
            st.markdown(f"""
            <div style="background:linear-gradient(145deg,#1c2333 0%,#161b22 100%);border:1px solid #30363d;border-radius:16px;padding:1.5rem;box-shadow:var(--shadow);">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
                <span style="font-size:0.75rem;font-weight:700;color:#8b949e;text-transform:uppercase;letter-spacing:0.08em;">OVERALL PERFORMANCE</span>
                <span style="font-size:0.75rem;font-weight:700;background:{tier_bg};color:{tier_color};border:1px solid {tier_color};border-radius:20px;padding:3px 12px;">{tier_label}</span>
              </div>
              <div style="display:flex;align-items:baseline;gap:0.5rem;margin:0.5rem 0;">
                <span style="font-size:3.5rem;font-weight:900;color:#ffffff;line-height:1;letter-spacing:-0.03em;">{overall}</span>
                <span style="font-size:1.25rem;font-weight:600;color:#8b949e;">/ 100</span>
              </div>
              <div style="background:#0d1117;border-radius:100px;height:10px;overflow:hidden;margin:1rem 0 0.75rem 0;border:1px solid #30363d;">
                <div style="width:{max(0, min(100, overall))}%;height:100%;background:linear-gradient(90deg,#1f6feb,#58a6ff,#3fb950);border-radius:100px;"></div>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#8b949e;">
                <span>Combined Index: <strong style="color:#e6edf3;">{combined_score}/100</strong> (75% Content + 25% Tone)</span>
                <span style="color:#58a6ff;font-weight:600;">Attempt #{latest_att.get('attempt_number', 1) if latest_att else 1}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with score_col2:
            st.markdown(f"""
            <div style="background:linear-gradient(145deg,#1c2333 0%,#161b22 100%);border:1px solid #30363d;border-radius:16px;padding:1.5rem;box-shadow:var(--shadow);">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
                <span style="font-size:0.75rem;font-weight:700;color:#8b949e;text-transform:uppercase;letter-spacing:0.08em;">ACOUSTIC TONE INTELLIGENCE</span>
                <span style="font-size:0.75rem;font-weight:700;background:rgba(88,166,255,0.12);color:#58a6ff;border:1px solid rgba(88,166,255,0.3);border-radius:20px;padding:3px 12px;">{model_used}</span>
              </div>
              <div style="display:flex;align-items:center;gap:1rem;margin:0.5rem 0;">
                <span style="font-size:3rem;line-height:1;">{tone_emoji}</span>
                <div>
                  <div style="font-size:1.8rem;font-weight:800;color:#ffffff;line-height:1.1;">{predicted_tone}</div>
                  <div style="font-size:0.82rem;color:{align_color};font-weight:600;margin-top:2px;">{align_badge} (Target: {st.session_state.expected_tone})</div>
                </div>
              </div>
              <div style="background:#0d1117;border-radius:100px;height:10px;overflow:hidden;margin:1rem 0 0.75rem 0;border:1px solid #30363d;">
                <div style="width:{max(0, min(100, int(tone_confidence*100)))}%;height:100%;background:linear-gradient(90deg,#bc8cff,#39d0d8);border-radius:100px;"></div>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#8b949e;">
                <span>Confidence: <strong style="color:#e6edf3;">{round(tone_confidence*100, 1)}%</strong></span>
                <span>Tone Score: <strong style="color:#39d0d8;">{tone_score}/100</strong></span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

        # ── 4 Core Dimensions Matrix ───────────────────────────────────────
        st.markdown('<div style="font-size:0.95rem;font-weight:700;color:#e6edf3;text-transform:uppercase;letter-spacing:0.06em;margin:1rem 0 0.75rem 0;">🎯 Core Speech Competency Matrix</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        m_rel = int(scores.get("topic_relevance", 0))
        m_cov = int(scores.get("key_point_coverage", 0))
        m_acc = int(scores.get("content_accuracy", 0))
        m_cla = int(scores.get("clarity", 0))

        with c1:
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid #30363d;border-top:3px solid #3fb950;border-radius:12px;padding:1rem;">
              <div style="font-size:0.75rem;color:#8b949e;text-transform:uppercase;font-weight:600;letter-spacing:0.05em;">🎯 Topic Relevance</div>
              <div style="font-size:1.8rem;font-weight:800;color:#3fb950;margin:0.25rem 0;">{m_rel}<span style="font-size:0.9rem;color:#8b949e;">/100</span></div>
              <div style="background:#0d1117;border-radius:100px;height:6px;overflow:hidden;margin-top:0.4rem;">
                <div style="width:{m_rel}%;height:100%;background:#3fb950;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid #30363d;border-top:3px solid #58a6ff;border-radius:12px;padding:1rem;">
              <div style="font-size:0.75rem;color:#8b949e;text-transform:uppercase;font-weight:600;letter-spacing:0.05em;">📚 Key Point Coverage</div>
              <div style="font-size:1.8rem;font-weight:800;color:#58a6ff;margin:0.25rem 0;">{m_cov}<span style="font-size:0.9rem;color:#8b949e;">/100</span></div>
              <div style="background:#0d1117;border-radius:100px;height:6px;overflow:hidden;margin-top:0.4rem;">
                <div style="width:{m_cov}%;height:100%;background:#58a6ff;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid #30363d;border-top:3px solid #bc8cff;border-radius:12px;padding:1rem;">
              <div style="font-size:0.75rem;color:#8b949e;text-transform:uppercase;font-weight:600;letter-spacing:0.05em;">🔍 Content Accuracy</div>
              <div style="font-size:1.8rem;font-weight:800;color:#bc8cff;margin:0.25rem 0;">{m_acc}<span style="font-size:0.9rem;color:#8b949e;">/100</span></div>
              <div style="background:#0d1117;border-radius:100px;height:6px;overflow:hidden;margin-top:0.4rem;">
                <div style="width:{m_acc}%;height:100%;background:#bc8cff;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid #30363d;border-top:3px solid #f0883e;border-radius:12px;padding:1rem;">
              <div style="font-size:0.75rem;color:#8b949e;text-transform:uppercase;font-weight:600;letter-spacing:0.05em;">🗣️ Articulation & Clarity</div>
              <div style="font-size:1.8rem;font-weight:800;color:#f0883e;margin:0.25rem 0;">{m_cla}<span style="font-size:0.9rem;color:#8b949e;">/100</span></div>
              <div style="background:#0d1117;border-radius:100px;height:6px;overflow:hidden;margin-top:0.4rem;">
                <div style="width:{m_cla}%;height:100%;background:#f0883e;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

        # ── Per-section score breakdown ───────────────────────────────────
        sections_with_scores = [s for s in st.session_state.sections if s.get("local_scores")]
        if sections_with_scores:
            st.markdown("""
            <div style="display:flex;align-items:center;justify-content:space-between;margin:1.25rem 0 0.75rem 0;">
              <span style="font-size:0.95rem;font-weight:700;color:#e6edf3;text-transform:uppercase;letter-spacing:0.06em;">🗂️ Section-by-Section Performance Breakdown</span>
              <span style="font-size:0.72rem;background:rgba(63,185,80,0.12);color:#3fb950;border-radius:12px;padding:2px 10px;font-weight:600;">Local Zero-Quota Scoring</span>
            </div>
            """, unsafe_allow_html=True)

            sec_cols = st.columns(len(sections_with_scores))
            for s_idx, sec in enumerate(sections_with_scores):
                with sec_cols[s_idx]:
                    sec_sc = sec["local_scores"].get("scores", {})
                    sec_overall = sec["local_scores"].get("overall_score", 0)
                    st.markdown(f"""
                    <div style="background:var(--bg-card);border:1px solid #30363d;border-radius:12px;padding:1rem;margin-bottom:0.5rem;">
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">
                        <span style="font-weight:700;color:#58a6ff;font-size:0.95rem;">{sec['name']}</span>
                        <span style="font-size:1.1rem;font-weight:800;color:#3fb950;">{sec_overall}<span style="font-size:0.75rem;color:#8b949e;">/100</span></span>
                      </div>
                      <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem;font-size:0.75rem;color:#8b949e;margin-top:0.6rem;">
                        <div>Relevance: <strong style="color:#e6edf3;">{sec_sc.get('topic_relevance', 0)}</strong></div>
                        <div>Coverage: <strong style="color:#e6edf3;">{sec_sc.get('key_point_coverage', 0)}</strong></div>
                        <div>Accuracy: <strong style="color:#e6edf3;">{sec_sc.get('content_accuracy', 0)}</strong></div>
                        <div>Clarity: <strong style="color:#e6edf3;">{sec_sc.get('clarity', 0)}</strong></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

        # ── Strengths & Growth Areas Bento Grid ───────────────────────────
        bento_col1, bento_col2 = st.columns(2)
        with bento_col1:
            strengths_list = "".join(f'<li style="margin-bottom:0.5rem;color:#e6edf3;font-size:0.875rem;">{s}</li>' for s in result.get("strengths", []))
            st.markdown(f"""
            <div style="background:linear-gradient(145deg,rgba(63,185,80,0.08) 0%,#161b22 100%);border:1px solid rgba(63,185,80,0.3);border-radius:14px;padding:1.3rem;height:100%;">
              <div style="display:flex;align-items:center;gap:0.5rem;font-weight:700;color:#3fb950;font-size:1rem;margin-bottom:0.75rem;">
                <span>🌟</span><span>Key Strengths</span>
              </div>
              <ul style="padding-left:1.2rem;margin:0;">{strengths_list}</ul>
            </div>
            """, unsafe_allow_html=True)

        with bento_col2:
            improvements_list = "".join(f'<li style="margin-bottom:0.5rem;color:#e6edf3;font-size:0.875rem;">{imp}</li>' for imp in result.get("improvements", []))
            st.markdown(f"""
            <div style="background:linear-gradient(145deg,rgba(240,136,62,0.08) 0%,#161b22 100%);border:1px solid rgba(240,136,62,0.3);border-radius:14px;padding:1.3rem;height:100%;">
              <div style="display:flex;align-items:center;gap:0.5rem;font-weight:700;color:#f0883e;font-size:1rem;margin-bottom:0.75rem;">
                <span>🚀</span><span>Growth Opportunities</span>
              </div>
              <ul style="padding-left:1.2rem;margin:0;">{improvements_list}</ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

        # ── Executive Summary Quote Block ──────────────────────────────────
        if result.get("final_summary"):
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(88,166,255,0.08) 0%,#161b22 100%);border:1px solid rgba(88,166,255,0.3);border-left:4px solid #58a6ff;border-radius:12px;padding:1.25rem 1.5rem;margin-bottom:1.5rem;">
              <div style="font-weight:700;color:#58a6ff;font-size:0.82rem;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.4rem;">💡 Executive Assessment Summary</div>
              <div style="color:#e6edf3;font-size:0.95rem;line-height:1.6;font-style:italic;">"{result.get('final_summary')}"</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Step-by-Step AI Breakdown ─────────────────────────────────────
        with st.expander("🔍 Deep-Dive: Step-by-Step Evaluation Logs", expanded=False):
            for item in result.get("step_by_step_analysis", []):
                st.markdown(f"""
                <div style="background:var(--bg-card);border:1px solid #30363d;border-radius:10px;padding:0.9rem 1.1rem;margin-bottom:0.6rem;">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem;">
                    <span style="font-weight:700;color:#e6edf3;font-size:0.9rem;">{item.get('step', 'Step')}</span>
                    <span style="font-size:0.82rem;font-weight:700;color:#58a6ff;background:rgba(88,166,255,0.15);padding:2px 8px;border-radius:6px;">{int(item.get('score', 0))}/100</span>
                  </div>
                  <div style="font-size:0.85rem;color:#8b949e;line-height:1.5;">{item.get('analysis', '')}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Evidence & Transcript Accordions ──────────────────────────────
        with st.expander("📄 Transcribed Speech Text", expanded=False):
            st.text_area("Transcript", value=st.session_state.transcript, height=140, disabled=True, label_visibility="collapsed")

        with st.expander("🎯 Retrieved RAG Evidence Passages", expanded=False):
            for i, passage in enumerate(st.session_state.retrieved_passages, start=1):
                st.markdown(f"**Passage {i} (Similarity: {passage['similarity']:.3f}):**\n\n{passage['text']}")

        if emotion.get("probabilities"):
            with st.expander("📊 Tone Class Probability Distribution", expanded=False):
                prob_map = emotion.get("probabilities", {})
                for label in ["Neutral", "Happy", "Angry", "Sad"]:
                    if label in prob_map:
                        st.write(f"- **{label}:** {float(prob_map[label]):.4f}")

        # Download report button
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Export Full Evaluation Report (JSON)",
            data=json.dumps(result, indent=2),
            file_name="speech_analysis_result.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    col_prev, col_reset = st.columns(2)
    with col_prev:
        if st.button("<- Back to Step 4", use_container_width=True):
            go_to_step(4)
            st.rerun()
    with col_reset:
        if st.button("🔄 Start New Speech Evaluation", type="primary", use_container_width=True):
            reset_all()
            st.rerun()


# ── Global Professional Footer ─────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding:2rem 0;border-top:1px solid #30363d;text-align:center;">
  <div style="display:flex;justify-content:center;gap:0.75rem;margin-bottom:0.75rem;flex-wrap:wrap;">
    <span style="font-size:0.72rem;background:#161b22;border:1px solid #30363d;color:#8b949e;padding:3px 10px;border-radius:20px;">OpenAI Whisper</span>
    <span style="font-size:0.72rem;background:#161b22;border:1px solid #30363d;color:#8b949e;padding:3px 10px;border-radius:20px;">FAISS Vector Engine</span>
    <span style="font-size:0.72rem;background:#161b22;border:1px solid #30363d;color:#8b949e;padding:3px 10px;border-radius:20px;">MiniLM-L6-v2 Embeddings</span>
    <span style="font-size:0.72rem;background:#161b22;border:1px solid #30363d;color:#8b949e;padding:3px 10px;border-radius:20px;">Google Gemini 2.0 Flash</span>
    <span style="font-size:0.72rem;background:#161b22;border:1px solid #30363d;color:#8b949e;padding:3px 10px;border-radius:20px;">MongoDB Atlas</span>
  </div>
  <div style="font-size:0.8rem;color:#8b949e;">SmartSpeak AI Enterprise Studio • Developed for High-Impact Speaking Performance</div>
</div>
""", unsafe_allow_html=True)