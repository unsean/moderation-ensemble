"""
Hate Speech Detector — Premium Streamlit Application
A modern, dark-mode SaaS-style web app for hate speech detection
using an ensemble of ML models (Stacking, SVM, Logistic Regression, Naive Bayes).
"""

import re
import json
import string
import pickle
from pathlib import Path

import streamlit as st
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# ─────────────────────────────────────────────────────
# 0. NLTK bootstrap (silent)
# ─────────────────────────────────────────────────────
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# ─────────────────────────────────────────────────────
# 1. Page config (MUST be first Streamlit command)
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hate Speech Detector",
    page_icon="./loto.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────
# 2. Inject custom CSS — premium dark-mode theme
# ─────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Import Google Fonts ────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── CSS Custom Properties ──────────────────────── */
:root {
    --bg-primary:    #0a0a0f;
    --bg-secondary:  #12121a;
    --bg-card:       #16161f;
    --bg-card-hover: #1c1c28;
    --bg-input:      #1a1a26;

    --border-subtle:  rgba(255, 255, 255, 0.06);
    --border-accent:  rgba(245, 197, 24, 0.35);

    --text-primary:   #f0f0f8;
    --text-secondary: #b0b0c4;
    --text-muted:     #8a8aa4;

    --accent:          #f5c518;
    --accent-light:    #fcd34d;
    --accent-glow:     rgba(245, 197, 24, 0.25);

    --safe-main:       #34d399;
    --safe-bg:         rgba(52, 211, 153, 0.08);
    --safe-border:     rgba(52, 211, 153, 0.25);
    --safe-glow:       rgba(52, 211, 153, 0.15);

    --danger-main:     #f87171;
    --danger-bg:       rgba(248, 113, 113, 0.08);
    --danger-border:   rgba(248, 113, 113, 0.25);
    --danger-glow:     rgba(248, 113, 113, 0.15);

    --warning-main:    #fbbf24;
    --warning-bg:      rgba(251, 191, 36, 0.08);
    --warning-border:  rgba(251, 191, 36, 0.25);

    --radius-sm: 8px;
    --radius-md: 14px;
    --radius-lg: 20px;

    --transition-fast: 0.18s cubic-bezier(.4,0,.2,1);
    --transition-med:  0.3s  cubic-bezier(.4,0,.2,1);
}

/* ── Global resets ──────────────────────────────── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* ── Sidebar ────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* Force bright text on Streamlit markdown */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li {
    color: var(--text-secondary) !important;
}

/* ── Text inputs / textareas ────────────────────── */
.stTextArea textarea,
.stTextInput input {
    background: var(--bg-input) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    transition: var(--transition-fast) !important;
    padding: 1rem !important;
}
.stTextArea textarea:focus,
.stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

/* ── Radio (used as tab bar) ────────────────────── */
div[data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: row !important;
    gap: 0 !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    padding: 4px !important;
    overflow-x: auto;
}
div[data-testid="stRadio"] label {
    flex: 1 1 0 !important;
    text-align: center !important;
    padding: 0.65rem 1.2rem !important;
    border-radius: 10px !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: var(--transition-fast) !important;
    white-space: nowrap !important;
    color: var(--text-secondary) !important;
    margin: 0 !important;
}
div[data-testid="stRadio"] label[data-checked="true"],
div[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, var(--accent), #e0a800) !important;
    color: #12121a !important;
    box-shadow: 0 2px 12px var(--accent-glow) !important;
}
/* Keep dark text even on hover for the active tab */
div[data-testid="stRadio"] label[data-checked="true"]:hover,
div[data-testid="stRadio"] label:has(input:checked):hover {
    color: #12121a !important;
    background: linear-gradient(135deg, var(--accent), #e0a800) !important;
}
/* Override Streamlit markdown containers inside active radio label */
div[data-testid="stRadio"] label[data-checked="true"] span,
div[data-testid="stRadio"] label:has(input:checked) span,
div[data-testid="stRadio"] label[data-checked="true"] p,
div[data-testid="stRadio"] label:has(input:checked) p,
div[data-testid="stRadio"] label[data-checked="true"] div,
div[data-testid="stRadio"] label:has(input:checked) div {
    color: #12121a !important;
}
div[data-testid="stRadio"] label:hover {
    color: var(--text-primary) !important;
    background: var(--bg-card-hover) !important;
}
/* Hide the radio circle */
div[data-testid="stRadio"] input[type="radio"] {
    display: none !important;
}
div[data-testid="stRadio"] label span[data-testid="stMarkdownContainer"] ~ div,
div[data-testid="stRadio"] label div[class*="indicator"],
div[data-testid="stRadio"] label > div:first-child {
    display: none !important;
}
/* Hide "Choose an option" label text for radio */
div[data-testid="stRadio"] > label {
    display: none !important;
}

/* ── Metric cards (for Model Information page) ──── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    padding: 1.4rem 1.6rem !important;
    transition: var(--transition-fast) !important;
}
[data-testid="stMetric"]:hover {
    border-color: var(--border-accent) !important;
    box-shadow: 0 4px 20px var(--accent-glow) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    font-size: 0.78rem !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent-light) !important;
    font-weight: 700 !important;
    font-size: 1.75rem !important;
}

/* ── Dataframe / Table ──────────────────────────── */
[data-testid="stDataFrame"], .stDataFrame {
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
}

/* ── Expander (NLP Insights) ────────────────────── */
details[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
}
details[data-testid="stExpander"] summary {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
}
details[data-testid="stExpander"] summary:hover {
    color: var(--text-primary) !important;
}

/* ── Code blocks (inside expander) ──────────────── */
code, pre {
    background: var(--bg-primary) !important;
    color: var(--accent-light) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Tabs (Model Information page) ──────────────── */
button[data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
}
button[data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
}
div[data-baseweb="tab-highlight"] {
    background-color: var(--accent) !important;
}

/* ── Utility classes (injected via st.markdown) ─── */
.card-container {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 2rem 2.2rem;
    margin-bottom: 1.5rem;
    transition: var(--transition-fast);
}
.card-container:hover {
    border-color: var(--border-accent);
    box-shadow: 0 6px 30px var(--accent-glow);
}
.card-safe {
    background: var(--safe-bg) !important;
    border-color: var(--safe-border) !important;
}
.card-safe:hover {
    box-shadow: 0 6px 30px var(--safe-glow) !important;
}
.card-danger {
    background: var(--danger-bg) !important;
    border-color: var(--danger-border) !important;
}
.card-danger:hover {
    box-shadow: 0 6px 30px var(--danger-glow) !important;
}

.page-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent-light), #fbbf24, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
    margin-bottom: 0.1rem;
}
.page-subtitle {
    color: var(--text-secondary);
    font-size: 1.05rem;
    font-weight: 400;
    margin-bottom: 2rem;
}

.result-label {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    margin-bottom: 0.4rem;
}
.result-label-safe   { color: var(--safe-main);   }
.result-label-danger { color: var(--danger-main); }
.result-label-warning{ color: var(--warning-main);}

.metric-mini {
    text-align: center;
}
.metric-mini .value {
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.1;
}
.metric-mini .label {
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
    margin-top: 0.3rem;
}

/* Donut chart wrapper */
.donut-wrapper {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding: 1rem 0;
}

/* ── Sidebar brand ──────────────────────────────── */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.3rem 0 1.5rem 0;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 1.5rem;
}
.sidebar-brand .icon {
    font-size: 1.6rem;
}
.sidebar-brand .name {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}

.sidebar-nav-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.7rem 1rem;
    border-radius: var(--radius-sm);
    margin-bottom: 0.3rem;
    font-weight: 500;
    font-size: 0.95rem;
    color: var(--text-secondary);
    cursor: pointer;
    transition: var(--transition-fast);
}
.sidebar-nav-item:hover {
    background: var(--bg-card-hover);
    color: var(--text-primary);
}
.sidebar-nav-item.active {
    background: var(--accent);
    color: #fff;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--text-muted); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }

/* Badge */
.badge {
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 99px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.badge-safe {
    background: var(--safe-bg);
    color: var(--safe-main);
    border: 1px solid var(--safe-border);
}
.badge-danger {
    background: var(--danger-bg);
    color: var(--danger-main);
    border: 1px solid var(--danger-border);
}
.badge-warning {
    background: var(--warning-bg);
    color: var(--warning-main);
    border: 1px solid var(--warning-border);
}

/* Hide Streamlit brand / footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 2.5rem !important;
}

/* Prevent sidebar drag-resize */
section[data-testid="stSidebar"] > div[data-testid="stSidebarResizeHandle"] {
    display: none !important;
}

/* Smooth fade-in animation */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.animate-in {
    animation: fadeInUp 0.45s var(--transition-fast) both;
}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────
# 3. Preprocessing pipeline (mirrors the notebook)
# ─────────────────────────────────────────────────────
CUSTOM_STOPWORDS = {
    "dont", "wont", "cant", "isnt", "arent", "wasnt", "werent",
    "hasnt", "havent", "hadnt", "doesnt", "didnt", "couldnt",
    "wouldnt", "shouldnt", "mightnt", "mustnt", "shan", "aint",
    "thats", "whats", "theres", "heres", "wheres", "whens",
    "whos", "whys", "hows", "shes", "hes", "its", "theyre",
    "weve", "youve", "ive", "youre", "theyve", "shouldve",
    "wouldve", "couldve", "gonna", "wanna", "gotta", "kinda",
    "sorta", "dunno", "lemme", "gimme", "coulda", "woulda",
    "shoulda", "oughta", "hafta", "tryna", "boutta", "ya",
    "yall", "yea", "yeah", "nah", "tho", "tht", "lol",
    "lmao", "lmfao", "omg", "smh", "tbh", "imo", "btw",
    "fyi", "idk", "ikr", "iirc", "afaik", "irl", "rn",
    "af", "ur", "u", "r", "b4", "gr8", "luv", "msg",
    "pls", "plz", "thx", "tx", "ty", "np", "yw",
    "im", "ill", "ive", "id", "youre", "youve", "youll",
    "youd", "hes", "hed", "hell", "shes", "shed", "shell",
    "its", "itd", "itll", "were", "weve", "well", "wed",
    "theyre", "theyve", "theyll", "theyd", "get", "got", "go",
    "going", "gone", "come", "came", "like", "just", "really",
    "even", "still", "also", "would", "could", "one", "much",
    "make", "made", "want", "need", "say", "said", "thing",
    "things", "people", "know", "think", "see", "look", "back",
    "way", "well", "right", "good", "new", "take", "took",
}

STOP_WORDS = set(stopwords.words("english")).union(CUSTOM_STOPWORDS)
stemmer = PorterStemmer()

# Homoglyph -> ASCII mapping (common evasion tricks)
HOMOGLYPH_MAP = str.maketrans("0123456789$@", "oizeasgtbqsa")


def normalize_homoglyphs(text: str) -> str:
    """Map leetspeak / homoglyph variants back to ASCII letters.
    Examples: n1gg3r -> nigger, f4ck -> fack, @ss -> ass.
    """
    return text.translate(HOMOGLYPH_MAP)


def preprocess_text(text: str) -> str:
    """Exact replica of the notebook preprocessing + homoglyph defence."""
    text = str(text).lower()
    text = normalize_homoglyphs(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    # First pass: 3+ repeats -> 2 (matches notebook)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    # Second pass: 2+ -> 1 (removes leetspeak duplicates after homoglyphs)
    text = re.sub(r"(.)\1+", r"\1", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = word_tokenize(text)
    processed_tokens = [
        stemmer.stem(t) for t in tokens if t not in STOP_WORDS and len(t) >= 2
    ]
    return " ".join(processed_tokens)


# ─────────────────────────────────────────────────────
# 4. Load artefacts (cached once per session)
# ─────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).resolve().parent / "models"

MODEL_KEYS = {
    "Stacking Ensemble": "stacking_ensemble",
    "SVM": "svm",
    "Logistic Regression": "logistic_regression",
    "Naive Bayes": "naive_bayes",
}


@st.cache_resource(show_spinner=False)
def load_vectorizer():
    with open(MODELS_DIR / "tfidf_vectorizer.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_model(key: str):
    with open(MODELS_DIR / f"{key}.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_training_summary():
    with open(MODELS_DIR / "training_summary.json", "r") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_explainer():
    """Load Logistic Regression for local explainability."""
    with open(MODELS_DIR / "logistic_regression.pkl", "rb") as f:
        return pickle.load(f)


def get_artifact_metadata():
    """Return metadata dict from artifact if available, else empty."""
    if artifact:
        return artifact.get("metadata", {})
    return {}


def clean_feature_name(name: str) -> str:
    """Strip FeatureUnion prefix and clean whitespace."""
    name = name.removeprefix("word__").removeprefix("char_wb__").removeprefix("char__")
    return name.strip()


def get_flagged_words(features, vectorizer, explainer, top_n=5):
    """Return the top TF-IDF features that pushed the prediction toward toxic."""
    if not hasattr(explainer, "coef_"):
        return []
    coef = explainer.coef_[0]
    row = features.tocsr()[0]
    indices = row.indices
    data = row.data

    contributions = [(idx, data[i] * coef[idx]) for i, idx in enumerate(indices)]
    contributions.sort(key=lambda x: x[1], reverse=True)

    feature_names = vectorizer.get_feature_names_out()
    seen = set()
    flagged = []
    for idx, score in contributions:
        if score <= 0:
            continue
        raw = feature_names[idx]
        clean = clean_feature_name(raw)
        if not clean or len(clean) < 2 or clean in seen:
            continue
        seen.add(clean)
        flagged.append((clean, float(score)))
        if len(flagged) >= top_n:
            break
    return flagged


ARTIFACT_PATH = MODELS_DIR / "deployment_artifact.pkl"


@st.cache_resource(show_spinner=False)
def load_artifact():
    """Load deployment artifact (model + vectorizer + metadata)."""
    if ARTIFACT_PATH.exists():
        with open(ARTIFACT_PATH, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_resource(show_spinner=False)
def load_vectorizer():
    artifact = load_artifact()
    if artifact:
        return artifact["vectorizer"]
    with open(MODELS_DIR / "tfidf_vectorizer.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_model(key: str):
    artifact = load_artifact()
    if artifact and key == "stacking_ensemble":
        return artifact["model"]
    with open(MODELS_DIR / f"{key}.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_training_summary():
    artifact = load_artifact()
    if artifact:
        meta = artifact["metadata"]
        # Build a training_summary-compatible dict from artifact metadata
        return {
            "best_model_name": meta.get("best_model_name", "Stacking Ensemble"),
            "f1_score": meta.get("best_model_metrics", {}).get("F1 Score", 0),
            "precision": meta.get("best_model_metrics", {}).get("Precision", 0),
            "recall": meta.get("best_model_metrics", {}).get("Recall", 0),
            "accuracy": meta.get("best_model_metrics", {}).get("Accuracy", 0),
            "all_models_results": meta.get("model_results", []),
        }
    with open(MODELS_DIR / "training_summary.json", "r") as f:
        return json.load(f)


# Pre-load everything at startup
artifact = load_artifact()
vectorizer = load_vectorizer()
training_summary = load_training_summary()
# Pre-warm default model cache so Analyze button feels instant
_default_model = load_model("stacking_ensemble")


# ─────────────────────────────────────────────────────
# 5. SVG donut-chart builder
# ─────────────────────────────────────────────────────
def _donut_svg(
    pct: float,
    label: str,
    color_main: str,
    color_track: str = "rgba(255,255,255,0.06)",
    size: int = 180,
    stroke: int = 14,
) -> str:
    """Return an SVG donut chart string."""
    radius = (size - stroke) / 2
    circumference = 2 * np.pi * radius
    dash = circumference * pct / 100
    gap = circumference - dash
    cx = cy = size / 2

    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}"'
        f' style="transform:rotate(-90deg)">'
        f'<circle cx="{cx}" cy="{cy}" r="{radius}"'
        f' fill="none" stroke="{color_track}" stroke-width="{stroke}" />'
        f'<circle cx="{cx}" cy="{cy}" r="{radius}"'
        f' fill="none" stroke="{color_main}" stroke-width="{stroke}"'
        f' stroke-dasharray="{dash} {gap}"'
        f' stroke-linecap="round"'
        f' style="transition: stroke-dasharray 0.6s ease;" />'
        f'<text x="{cx}" y="{cx}" text-anchor="middle" dominant-baseline="central"'
        f' fill="{color_main}" font-size="2rem" font-weight="800"'
        f' font-family="Inter, sans-serif"'
        f' style="transform:rotate(90deg);transform-origin:center;">'
        f'{pct:.1f}%</text></svg>'
        f'<p style="text-align:center;margin-top:0.5rem;">'
        f'<span style="color:{color_main};font-weight:600;font-size:1.25rem;'
        f'letter-spacing:0.03em;">{label}</span></p>'
    )


# ─────────────────────────────────────────────────────
# 6. Sidebar navigation
# ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <span class="icon"></span>
            <span class="name">ModGuard</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "nav",
        ["Hate Speech Detector", "Model Information"],
        label_visibility="collapsed",
    )


# ─────────────────────────────────────────────────────
# 7. PAGE 1 — Hate Speech Detector
# ─────────────────────────────────────────────────────
if page == "Hate Speech Detector":

    # Title
    st.markdown(
        """
        <div class="animate-in">
            <p class="page-title">Hate Speech Detector</p>
            <p class="page-subtitle">
                Analyze text for toxic or hateful content using state-of-the-art
                classical ML models trained on 726,119 samples.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Model selector tab bar ──────────────────────
    model_names = list(MODEL_KEYS.keys())
    selected_model = st.radio(
        "Select Model",
        model_names,
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

    st.write("")  # spacer

    # ── Input area ──────────────────────────────────
    with st.form(key="analyze_form", enter_to_submit=True):
        user_text = st.text_area(
            "input_text",
            placeholder="Type or paste content to analyze…",
            height=170,
            label_visibility="collapsed",
        )
        analyze_clicked = st.form_submit_button(
            "Analyze Text", use_container_width=True
        )

    # ── Analysis logic ──────────────────────────────
    if analyze_clicked:
        if not user_text.strip():
            st.warning("Please enter some text before analyzing.")
        else:
            with st.spinner("Analyzing…"):
                model_key = MODEL_KEYS[selected_model]
                model = load_model(model_key)

                preprocessed = preprocess_text(user_text)
                features = vectorizer.transform([preprocessed])

                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]

                # probabilities: [p_safe, p_hate]
                safe_pct = probabilities[0] * 100
                hate_pct = probabilities[1] * 100

                is_hate = prediction == 1
                flagged_words = []
                if is_hate:
                    explainer = load_explainer()
                    flagged_words = get_flagged_words(features, vectorizer, explainer)

                # ── Results card ────────────────────────
            card_class = "card-danger" if is_hate else "card-safe"
            verdict_color = "var(--danger-main)" if is_hate else "var(--safe-main)"
            verdict_text = "Hate Speech Detected" if is_hate else "Text Appears Safe"
            _toxic_icon = (
                '<span style="display:inline-flex;align-items:center;'
                'justify-content:center;width:52px;height:52px;border-radius:50%;'
                'background:var(--danger-bg);">'
                '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" '
                'viewBox="0 0 24 24" fill="none" stroke="var(--danger-main)" '
                'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
                '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
                '<line x1="12" y1="9" x2="12" y2="13"/>'
                '<line x1="12" y1="17" x2="12.01" y2="17"/>'
                '</svg></span>'
            )
            _safe_icon = (
                '<span style="display:inline-flex;align-items:center;'
                'justify-content:center;width:52px;height:52px;border-radius:50%;'
                'background:var(--safe-bg);">'
                '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" '
                'viewBox="0 0 24 24" fill="none" stroke="var(--safe-main)" '
                'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
                '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'
                '<polyline points="9 12 12 15 16 11"/>'
                '</svg></span>'
            )
            verdict_icon = _toxic_icon if is_hate else _safe_icon
            badge_class = "badge-danger" if is_hate else "badge-safe"

            st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="card-container {card_class} animate-in"
                     style="padding:1.6rem 2rem;">
                    <div style="display:flex;align-items:center;gap:1.2rem;">
                        {verdict_icon}
                        <span class="result-label"
                              style="color:{verdict_color};font-size:1.5rem;">
                            {verdict_text}
                        </span>
                        <span class="badge {badge_class}"
                              style="margin-left:auto;font-size:0.95rem;
                                     padding:0.45rem 1.1rem;">
                            {selected_model}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Donut charts ────────────────────────
            col_safe, col_hate = st.columns(2)

            with col_safe:
                safe_donut = _donut_svg(safe_pct, "Safe / Non-toxic", "#34d399")
                safe_html = f'<div class="card-container animate-in" style="animation-delay:0.08s;"><div class="donut-wrapper">{safe_donut}</div></div>'
                st.markdown(safe_html, unsafe_allow_html=True)

            with col_hate:
                hate_color = "#f87171" if hate_pct >= 50 else "#fbbf24"
                hate_donut = _donut_svg(hate_pct, "Hate Speech", hate_color)
                hate_html = f'<div class="card-container animate-in" style="animation-delay:0.14s;"><div class="donut-wrapper">{hate_donut}</div></div>'
                st.markdown(hate_html, unsafe_allow_html=True)

            # ── Confidence bar (extra polish) ───────
            if is_hate:
                bar_fill_style = (
                    f"width:{hate_pct}%;height:100%;border-radius:99px;"
                    f"background:linear-gradient(90deg,#f87171,#ef4444);"
                    f"margin-left:auto;transition:width 0.6s ease;"
                )
            else:
                bar_fill_style = (
                    f"width:{safe_pct}%;height:100%;border-radius:99px;"
                    f"background:linear-gradient(90deg,#34d399,#2dd4bf);"
                    f"transition:width 0.6s ease;"
                )

            st.markdown(
                f"""
                <div class="card-container animate-in"
                     style="animation-delay:0.2s;">
                    <p style="color:var(--text-secondary);font-weight:600;
                              font-size:0.82rem;text-transform:uppercase;
                              letter-spacing:0.06em;margin-bottom:0.8rem;">
                        Confidence Breakdown
                    </p>
                    <div style="display:flex;align-items:center;gap:1rem;">
                        <span style="color:#34d399;font-weight:700;
                                     min-width:60px;">
                            {safe_pct:.1f}%
                        </span>
                        <div style="flex:1;height:10px;border-radius:99px;
                                    background:rgba(255,255,255,0.06);
                                    overflow:hidden;">
                            <div style="{bar_fill_style}">
                            </div>
                        </div>
                        <span style="color:{hate_color};font-weight:700;
                                     min-width:60px;text-align:right;">
                            {hate_pct:.1f}%
                        </span>
                    </div>
                    <div style="display:flex;justify-content:space-between;
                                margin-top:0.4rem;">
                        <span style="color:var(--text-muted);
                                     font-size:0.75rem;">Safe</span>
                        <span style="color:var(--text-muted);
                                     font-size:0.75rem;">Hate</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Flagged Words (only when hate speech detected)
            if is_hate and flagged_words:
                flagged_items = " ".join(
                    f'''<span class="badge badge-danger" style="margin:0.25rem;display:inline-block;">{word}</span>'''
                    for word, _ in flagged_words
                )
                st.markdown(
                    f'''
                    <div class="card-container card-danger animate-in"
                         style="animation-delay:0.25s;">
                        <p style="color:var(--text-secondary);font-weight:600;
                                  font-size:0.82rem;text-transform:uppercase;
                                  letter-spacing:0.06em;margin-bottom:0.8rem;">
                            Flagged Keywords
                        </p>
                        <div style="display:flex;flex-wrap:wrap;gap:0.4rem;">
                            {flagged_items}
                        </div>
                        <p style="color:var(--text-muted);font-size:0.75rem;
                                  margin-top:0.6rem;">
                            These words / phrases had the strongest positive
                            contribution toward the hate-speech prediction.
                        </p>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )

            # ── NLP Insights (collapsible) ──────────
            with st.expander("NLP Insights — Preprocessed Text"):
                st.code(preprocessed if preprocessed else "(empty after preprocessing)", language="text")
                st.markdown(
                    f"""
                    <p style="color:var(--text-muted);font-size:0.8rem;
                              margin-top:0.5rem;">
                        The raw input was lowercased, stripped of URLs / mentions /
                        punctuation, normalized for repeated characters, tokenized,
                        stop-word filtered, and Porter-stemmed before being
                        vectorized with TF-IDF.
                    </p>
                    """,
                    unsafe_allow_html=True,
                )

# ─────────────────────────────────────────────────────
# 8. PAGE 2 — Model Information
# ─────────────────────────────────────────────────────
elif page == "Model Information":

    st.markdown(
        """
        <div class="animate-in">
            <p class="page-title">Model Information</p>
            <p class="page-subtitle">
                Global performance metrics for every model in the pipeline,
                evaluated on a held-out test set of ~145 k samples.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Best model highlight ────────────────────────
    best_name = training_summary["best_model_name"]
    best_f1 = training_summary["f1_score"]
    best_acc = training_summary["accuracy"]
    best_prec = training_summary["precision"]
    best_rec = training_summary["recall"]

    st.markdown(
        f"""
        <div class="card-container animate-in"
             style="border-color:var(--accent);
                    background:linear-gradient(135deg,
                    rgba(245,197,24,0.08), rgba(252,211,77,0.04));">
            <div style="display:flex;align-items:center;gap:0.7rem;
                        margin-bottom:0.8rem;">
                <span style="font-size:1.3rem;"></span>
                <span style="font-weight:700;font-size:1.1rem;
                             color:var(--accent-light);">
                    Best Model — {best_name}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick top-level metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("F1 Score", f"{best_f1 * 100:.2f}%")
    m2.metric("Accuracy", f"{best_acc * 100:.2f}%")
    m3.metric("Precision", f"{best_prec * 100:.2f}%")
    m4.metric("Recall", f"{best_rec * 100:.2f}%")

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── Per-model detail cards ──────────────────────
    all_results = training_summary["all_models_results"]

    # Build tabs for each model
    tab_labels = [r["Model"] for r in all_results]
    tabs = st.tabs(tab_labels)

    for tab, result in zip(tabs, all_results):
        with tab:
            # Metrics row
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy",  f"{result['Accuracy'] * 100:.2f}%")
            c2.metric("Precision", f"{result['Precision'] * 100:.2f}%")
            c3.metric("Recall",    f"{result['Recall'] * 100:.2f}%")
            c4.metric("F1 Score",  f"{result['F1 Score'] * 100:.2f}%")
            c5.metric("ROC-AUC",   f"{result['ROC-AUC'] * 100:.2f}%")

            st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

            # Cross-validation info
            cv_mean = result.get("CV F1 Mean", 0)
            cv_std = result.get("CV F1 Std", 0)

            st.markdown(
                f"""
                <div class="card-container animate-in">
                    <p style="color:var(--text-secondary);font-weight:600;
                              font-size:0.82rem;text-transform:uppercase;
                              letter-spacing:0.06em;margin-bottom:0.6rem;">
                        Cross-Validation (5-Fold Stratified)
                    </p>
                    <div style="display:flex;gap:2rem;">
                        <div class="metric-mini">
                            <div class="value"
                                 style="color:var(--accent-light);">
                                {cv_mean * 100:.2f}%
                            </div>
                            <div class="label">Mean F1</div>
                        </div>
                        <div class="metric-mini">
                            <div class="value"
                                 style="color:var(--text-primary);">
                                ±{cv_std * 100:.2f}%
                            </div>
                            <div class="label">Std Dev</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Full comparison table ───────────────────────
    st.markdown(
        """
        <div class="animate-in" style="animation-delay:0.15s;">
            <p style="font-weight:700;font-size:1.1rem;
                      margin-bottom:0.6rem;">
                Full Comparison Table
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    import pandas as pd

    df = pd.DataFrame(all_results)
    display_cols = ["Model", "F1 Score", "Precision", "Recall", "Accuracy", "ROC-AUC", "CV F1 Mean", "CV F1 Std"]
    df = df[display_cols].sort_values("F1 Score", ascending=False).reset_index(drop=True)

    # Format numeric columns as percentages
    for col in display_cols[1:]:
        df[col] = df[col].map(lambda x: f"{x * 100:.2f}%")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )
