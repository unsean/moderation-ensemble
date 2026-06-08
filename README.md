# Hate Speech Detection with Ensemble Machine Learning

> A deployed web application that automatically detects hate speech and toxic language in English text using classical machine learning with an ensemble approach.

**Live Demo:** [ensemble-ml-jmw.streamlit.app](https://ensemble-ml-jmw.streamlit.app)

---

## 1. Project Description

This project builds an end-to-end hate speech detection system. The goal is to classify whether a given text contains toxic or hateful content. The system combines Natural Language Processing (NLP) techniques with classical machine learning models and is deployed as an interactive web application.

**Key Features:**
- Full NLP preprocessing pipeline (lowercasing, stemming, stopword removal, repeated character normalization)
- TF-IDF feature extraction using FeatureUnion (word + character n-grams)
- Stacking Ensemble model combining Logistic Regression, Naive Bayes, and SVM
- Interactive Streamlit web app with confidence scores, flagged keywords, and model comparison
- User testing for usability and usefulness evaluation

**Tech Stack:** Python, scikit-learn, NLTK, Streamlit, Pandas, NumPy

---

## 2. Dataset

| Property | Value |
|----------|-------|
| **Name** | HateSpeechDatasetBalanced.csv |
| **Size** | 726,119 rows, 2 columns |
| **Columns** | `Content` (text), `Label` (0=Non-Toxic, 1=Toxic) |
| **Distribution** | 50.2% Toxic, 49.8% Non-Toxic (balanced) |
| **After Preprocessing** | 723,890 rows (3,229 empty rows removed) |

**Dataset Notes:**
- The dataset is balanced, preventing model bias toward one class.
- Contains 25,046 duplicate rows that were identified during EDA.
- Likely sourced from Wikipedia talk pages, introducing domain-specific vocabulary (e.g., "article", "page", "edit").
- File size is 138 MB; tracked via Git LFS due to GitHub's 100 MB file limit.

---

## 3. Analysis

The complete analysis — including Exploratory Data Analysis (EDA), preprocessing, feature extraction, model training, evaluation, and visualization — is documented in **`main.ipynb`**.

### Summary of Findings

**EDA Highlights:**
- Toxic texts are shorter on average (149 chars) compared to non-toxic texts (245 chars).
- Top toxic words include `fuck`, `slut`, `bitch`, `shit`, `ass`.
- Top non-toxic words include `articl`, `page`, `edit`, `wikipedia`, `thank`.
- Clear vocabulary separation exists between the two classes.

**Preprocessing Pipeline (12 steps):**
Lowercase → URL/HTML/Mention/Hashtag removal → Repeated character normalization → Punctuation removal → Non-alphabetic removal → Tokenization → Stopword removal → Porter stemming → Short token removal → Empty text removal.

**Feature Extraction:**
- `FeatureUnion` combining:
  - **Word TF-IDF:** 1-2 grams, 12,000 features
  - **Char TF-IDF:** 3-4 grams (word-boundary), 8,000 features
- **Total features:** 20,000

**Model Results:**

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Stacking Ensemble** | **0.8276** | **0.8277** | **0.8276** | **0.8276** | 0.9098 |
| SVM | 0.8275 | 0.8276 | 0.8275 | 0.8275 | 0.9099 |
| Logistic Regression | 0.8236 | 0.8237 | 0.8236 | 0.8236 | 0.9062 |
| Naive Bayes | 0.7781 | 0.7844 | 0.7781 | 0.7768 | 0.8626 |

**Validation:**
- 5-Fold Stratified Cross-Validation confirms no overfitting.
- CV F1 (0.8268) ≈ Test F1 (0.8276), indicating good generalization.

---

## 4. Code

| File | Description |
|------|-------------|
| `main.ipynb` | Complete ML pipeline: EDA, preprocessing, training, evaluation, model saving |
| `app.py` | Streamlit web application for real-time predictions |
| `models/` | Trained models, vectorizer, and metadata (.pkl files) |
| `requirements.txt` | Python dependencies |

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Run the app
streamlit run app.py
```

### Project Structure

```
.
├── data/
│   └── HateSpeechDatasetBalanced.csv
├── models/
│   ├── stacking_ensemble.pkl
│   ├── svm.pkl
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── model_metadata.json
│   └── training_summary.json
├── app.py
├── main.ipynb
├── requirements.txt
└── README.md
```

---

## 5. Deployment Link

**Live Application:** [https://ensemble-ml-jmw.streamlit.app](https://ensemble-ml-jmw.streamlit.app)

The deployed application allows users to:
- Input English text for real-time toxicity analysis
- Select from 4 trained models for prediction comparison
- View prediction results with color-coded indicators (red/green)
- See confidence scores and toxicity percentage breakdown
- Identify flagged keywords that contributed to the prediction
- Access model performance metrics and comparison charts

---

## Limitations & Future Work

- **Language:** Model only supports English text.
- **Context:** TF-IDF is a bag-of-words approach and does not understand sentence context or word order (e.g., "not bad" vs "bad").
- **Sarcasm:** Implicit toxicity and sarcasm remain challenging to detect.
- **Dataset Bias:** The dataset appears domain-specific (likely Wikipedia talk pages), which may affect generalization to other platforms like Twitter or TikTok.
- **User Testing:** Respondents were limited to students aged 18-24, not fully representative of all demographics.

**Future Improvements:**
- Multilingual support using multilingual BERT or similar models
- Larger and more diverse dataset from multiple platforms
- Enhanced explanation quality for ensemble predictions
- User feedback loop for continuous model improvement
- Testing with more diverse user groups

---

## License

This project was developed for academic purposes.
