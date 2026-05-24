import re
import string
import pickle
import json
from pathlib import Path

import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

MODELS_DIR = Path("models")

st.set_page_config(page_title="Hate Speech Detector", page_icon="shield")


# ─── Custom Stopwords (same as notebook) ─────────────────────────────────────
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
    "theyre", "theyve", "theyll", "theyd", "get", "got", "go", "going", "gone", "come", "came",
    "like", "just", "really", "even", "still", "also", "would", "could",
    "one", "much", "make", "made", "want", "need", "say", "said",
    "thing", "things", "people", "know", "think", "see", "look",
    "back", "way", "well", "right", "good", "new", "take", "took",
}

STOP_WORDS = set(stopwords.words('english')).union(CUSTOM_STOPWORDS)
stemmer = PorterStemmer()


def preprocess_text(text):
    """Preprocess text (same as notebook)."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = word_tokenize(text)
    processed_tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) >= 2
    ]
    return ' '.join(processed_tokens)


@st.cache_resource
def load_model():
    """Load the ensemble model, vectorizer, and training summary."""
    with open(MODELS_DIR / "stacking_ensemble.pkl", 'rb') as f:
        model = pickle.load(f)

    with open(MODELS_DIR / "tfidf_vectorizer.pkl", 'rb') as f:
        vectorizer = pickle.load(f)

    with open(MODELS_DIR / "training_summary.json", 'r') as f:
        summary = json.load(f)

    return model, vectorizer, summary


model, vectorizer, summary = load_model()


def predict(text):
    """Run prediction on a single text."""
    processed = preprocess_text(text)
    features = vectorizer.transform([processed])
    prediction = model.predict(features)[0]

    toxic_prob = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        toxic_prob = proba[1]

    return prediction, toxic_prob, processed


# ─── UI ──────────────────────────────────────────────────────────────────────

st.title("Hate Speech Detector")

# Model performance
st.header("Model: Stacking Ensemble")
res = next(
    (r for r in summary.get("all_models_results", []) if r["Model"] == "Stacking Ensemble"),
    None,
)
if res:
    cols = st.columns(5)
    cols[0].metric("F1 Score", f"{res['F1 Score']:.2f}")
    cols[1].metric("Accuracy", f"{res['Accuracy']:.2f}")
    cols[2].metric("Precision", f"{res['Precision']:.2f}")
    cols[3].metric("Recall", f"{res['Recall']:.2f}")
    cols[4].metric("ROC-AUC", f"{res['ROC-AUC']:.2f}")

# Main input
with st.form(key="analyze_form", enter_to_submit=True):
    user_text = st.text_input("Enter text to analyze:", placeholder="Type or paste text here...")
    analyze_btn = st.form_submit_button("Analyze", type="primary", use_container_width=True)

if analyze_btn:
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            prediction, toxic_prob, processed = predict(user_text)

        if prediction == 1:
            st.error("**TOXIC — Hate Speech Detected**")
        else:
            st.success("**NON-TOXIC**")

        if toxic_prob is not None:
            col1, col2 = st.columns(2)
            col1.metric("Toxic Probability", f"{toxic_prob:.2%}")
            col2.metric("Confidence", f"{max(toxic_prob, 1 - toxic_prob):.2%}")

        with st.expander("Preprocessed Text"):
            st.code(processed)
