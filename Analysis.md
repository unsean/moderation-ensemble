# Analysis

## 1. Summary

This project builds an end-to-end **English hate speech / toxic text detection** system using **classical machine learning** and a **Streamlit web application**. The goal is to classify user-generated English text into two classes:

- **0** → Non-Toxic / Safe
- **1** → Toxic / Hate Speech

Our project follows a complete machine learning workflow:

1. Load and inspect a large labeled text dataset  
2. Clean and normalize noisy social media text  
3. Convert text into numerical features using a hybrid TF-IDF representation  
4. Train and compare multiple classical ML models  
5. Select the best-performing model using F1-score  
6. Save the trained model and artifacts for deployment  
7. Expose the system through a user-facing web application  
8. Evaluate the app through real user testing  

---

## 2. Problem Statement

Online platforms generate a very large volume of comments, many of which may contain toxic or hateful language. Manual moderation is slow, inconsistent, and difficult to scale. A machine learning system can help by automatically detecting whether a piece of English text is likely to be toxic or non-toxic.

The task is naturally framed as a **binary text classification** problem:

- **Input:** English comment / message / social media text  
- **Output:** Toxic or Non-Toxic label  

This formulation is appropriate because the target is not sentiment in general, but moderation-oriented classification.

---

## 3. Dataset Analysis

### 3.1 Dataset identity

The project uses the **Hate Speech Detection Curated Dataset** from Kaggle, with an academic source DOI shown in the presentation:

- **DOI:** `10.1038/s41597-022-01864-y`

The dataset used in the project contains:

- **Raw dataset size:** 726,119 samples  
- **Processed dataset size:** 723,890 samples  
- **Task type:** Binary classification  
- **Label 0:** Non-Toxic  
- **Label 1:** Toxic / Hate Speech  

### 3.2 Class distribution

The dataset is close to balanced:

- **Non-Toxic:** 361,594  
- **Toxic:** 364,525  

This is a strong design choice because it reduces the risk of a majority-class bias. In a heavily imbalanced setting, a model can look good in accuracy while still failing to detect toxic text properly. Here, the label balance makes evaluation more meaningful.

### 3.3 Dataset suitability

This dataset is suitable for the project because:

- it contains English social media text,
- it already has toxicity labels,
- it is structured and tabular,
- it supports supervised learning,
- it is large enough to train classical ML models reliably.

### 3.4 Data quality observations

The project analysis highlights several useful observations:

- the dataset contains duplicate rows,
- the language is English only,
- the text style is noisy and informal,
- the vocabulary is domain-specific and social-media-like.

These properties justify the preprocessing and feature-engineering choices used later in the pipeline.

---

## 4. Text Preprocessing

The raw text is noisy and cannot be fed directly into classical ML models. The project therefore applies a custom preprocessing pipeline.

### 4.1 Preprocessing steps

The notebook applies the following steps:

1. Convert to lowercase  
2. Remove URLs  
3. Remove HTML tags  
4. Remove mentions and hashtags  
5. Normalize repeated characters  
6. Remove punctuation  
7. Remove non-alphabetic characters  
8. Tokenize the text  
9. Remove stopwords  
10. Apply Porter stemming  
11. Remove empty or invalid text  

### 4.2 Custom stopword design

A custom stopword list is added on top of the standard English stopwords. This includes:

- contractions without apostrophes,
- slang and social-media abbreviations,
- conversational fillers,
- generic high-frequency words that do not add much signal.

This is important because hate speech detection on social media often contains informal language, abbreviations, and noisy expressions that standard stopword lists do not handle well.

### 4.3 Why this preprocessing matters

The preprocessing pipeline reduces noise, simplifies the vocabulary, and helps the model focus on signal-bearing words. For hate speech detection, this is critical because toxic language is often expressed with:

- spelling variations,
- elongated words,
- slang,
- partial obfuscation,
- and informal syntax.

---

## 5. Feature Engineering

The project does not rely on raw text. Instead, it transforms text into numerical vectors using a **hybrid TF-IDF FeatureUnion**.

### 5.1 Word-level TF-IDF

The word-level vectorizer uses:

- `ngram_range=(1, 2)`
- `max_features=12000`
- `sublinear_tf=True`
- `min_df=5`
- `max_df=0.95`

Word n-grams are useful for capturing semantic content such as:
- individual toxic terms,
- short phrases,
- contextual word combinations.

### 5.2 Character-level TF-IDF

The character-level vectorizer uses:

- `analyzer='char_wb'`
- `ngram_range=(3, 4)`
- `max_features=8000`
- `sublinear_tf=True`
- `min_df=5`
- `max_df=0.95`

Character n-grams are useful for:
- misspellings,
- slang,
- obfuscated toxic words,
- morphological patterns.

### 5.3 Why combine word and character TF-IDF?

The combination is strong for hate speech detection because the two feature types complement each other:

- **word features** capture meaning and phrase-level semantics,
- **character features** capture noisy variants and spelling manipulations.

This design is especially useful for toxic text, where users often intentionally distort words.

### 5.4 Feature size

The final TF-IDF matrix contains:

- **12,000 word-level features**
- **8,000 character-level features**
- **20,000 total features**

This is a high-dimensional but standard and effective representation for classical NLP.

---

## 6. Experimental Setup

### 6.1 Train-test split

The data is split into:

- **80% training**
- **20% testing**

This is a standard setup for evaluating generalization on unseen data.

### 6.2 Cross-validation

The project uses **5-fold stratified cross-validation** to obtain more stable estimates of model performance. Stratification is important because it preserves the class distribution across folds.

### 6.3 Evaluation metrics

The project uses:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

F1-score is the most important metric here because hate speech detection is sensitive to both false positives and false negatives.

---

## 7. Models Trained

The project trains and compares the following models:

### 7.1 Logistic Regression

A strong linear baseline for text classification.  
The notebook uses a balanced class weight setting and a reasonably large iteration limit to ensure convergence.

### 7.2 Naive Bayes

A simple probabilistic baseline, often useful for text tasks because of its speed and robustness.

### 7.3 SVM

Implemented with a calibrated linear SVM approach so that probability-like outputs are available. This is appropriate for high-dimensional TF-IDF features.

### 7.4 Stacking Ensemble

The final ensemble combines:
- Logistic Regression
- Naive Bayes
- SVM

The meta-classifier is also Logistic Regression.  
This lets the final predictor learn how to weight the base models’ outputs.

---

## 8. Training and Evaluation Results

The final model comparison in the notebook is:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Stacking Ensemble | 82.76% | 82.77% | 82.76% | **82.76%** | 90.98% |
| SVM | 82.75% | 82.76% | 82.75% | 82.75% | 86.26% |
| Logistic Regression | 82.36% | 82.37% | 82.36% | 82.36% | 86.26% |
| Naive Bayes | 77.81% | 78.44% | 77.81% | 77.68% | 86.26% |

### 8.1 Main takeaway

The **Stacking Ensemble** achieved the best F1-score and was selected as the final model.

### 8.2 Why the result makes sense

This result is consistent with the nature of the problem:

- Logistic Regression and SVM are both strong linear text classifiers.
- Naive Bayes acts as a simpler probabilistic complement.
- Stacking can combine their different decision patterns and slightly improve overall performance.

### 8.3 Best model decision

The notebook selects the best model based on **F1-score**, which is a sensible choice for moderation tasks.

---

## 9. Detailed Evaluation of the Best Model

The notebook also generates:

- a detailed classification report,
- a confusion matrix,
- a heatmap visualization.

These outputs help answer practical questions such as:

- how many toxic examples were detected correctly,
- how many non-toxic texts were falsely flagged,
- whether the model is better at precision or recall.

This is important because moderation systems are not evaluated only by a single metric; the type of error also matters.

---

## 10. Deployment and Inference Flow

The project does not stop at training. The trained model is saved and later used in a web application.

### 10.1 Inference flow

The deployed prediction flow is:

1. user enters text,
2. text is preprocessed,
3. text is transformed using the saved TF-IDF vectorizer,
4. the selected model predicts the class,
5. the application returns:
   - toxic / non-toxic label,
   - confidence score,
   - explanatory output.

### 10.2 Deployment artifact

The notebook verifies that the saved deployment artifact can be reloaded correctly and used for prediction. This is important because it shows the project is deployment-ready, not only notebook-ready.

---

## 11. Application Analysis

The presentation shows that the web application includes:

- a text input box,
- model selection tabs,
- hate speech / safe prediction,
- safe and hate speech percentages,
- confidence breakdown,
- flagged keywords,
- preprocessed text insight,
- a model information page with benchmark results.

This is a strong application design because it does not only output a label. It also gives supporting information that helps a user understand why a prediction was produced.

### 11.1 Why this matters

For moderation tasks, interpretability is useful because users want to know:
- what model is doing,
- how confident it is,
- and what parts of the input might have contributed to the result.

---

## 12. User Testing Analysis

The presentation includes real user testing results.

### 12.1 Testing design

The app was tested by:

- **8 respondents**
- all outside the project team
- using Google Forms
- with both quantitative and qualitative feedback

### 12.2 Usability results

Average scores reported in the presentation show strong usability:

- Easy to use: **4.62 / 5**
- Interface clear and organized: **4.62 / 5**
- Input and analyze button easy to understand: **4.50 / 5**
- Prediction result easy to understand: **4.50 / 5**
- Application responds quickly: **4.38 / 5**
- Overall application rating: **4.38 / 5**

### 12.3 Feature usefulness results

The most appreciated features were:

- color indicator,
- confidence score,
- model information page,
- toxicity percentage display,
- flagged keywords.

The average trust in prediction result is also reasonably positive.

### 12.4 Qualitative feedback themes

Positive feedback included:
- simple and clean design,
- ability to test different models,
- useful model selection,
- confidence score,
- clear interface.

Improvement suggestions included:
- faster processing,
- support for abbreviations or other languages,
- more features,
- better handling of mixed-positive-negative sentences,
- more data diversity.

### 12.5 Recommendation result

The presentation shows:
- **Yes:** 6
- **Maybe:** 2
- **No:** 0

This is a strong sign that the application is useful from a user perspective.

---

## 13. Strengths of the Project

This project has several strengths:

### 13.1 End-to-end pipeline
It covers the full path from raw text to deployed prediction.

### 13.2 Strong feature engineering
The hybrid TF-IDF design is well matched to noisy social media text.

### 13.3 Model comparison
Multiple models were benchmarked instead of relying on a single classifier.

### 13.4 Clear final model selection
The best model was selected using F1-score, which is appropriate for this task.

### 13.5 Real deployment
The project includes a usable web application rather than only offline evaluation.

### 13.6 Real user feedback
The user testing section demonstrates practical validation beyond metrics.

---

## 14. Limitations

The project also has limitations that should be acknowledged:

- it focuses only on English text,
- it may struggle with sarcasm or implicit toxicity,
- context-dependent comments can be hard to classify,
- false positives and false negatives are still possible,
- the model can inherit bias from the training data,
- user testing is useful but still limited in scale.

These limitations do not weaken the project; they show realistic understanding of what the system can and cannot do.

---

## 15. Final Assessment

Overall, this is a solid classical machine learning project with a clear moderation use case. The project has:

- a well-defined binary classification task,
- a large labeled dataset,
- strong preprocessing,
- thoughtful feature extraction,
- multiple models for comparison,
- a final stacking ensemble,
- a deployed application,
- and user testing with useful feedback.

The main technical conclusion is that **Stacking Ensemble performed best**, with an F1-score of **82.76%**, making it the final model for the hate speech detection system.

The main product conclusion is that the application is usable, understandable, and useful as a moderation support tool, even though it still has natural limitations common to text classification systems.
