# app.py
import streamlit as st
import pandas as pd
import os
import re
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------- DEFAULT PATHS (from this session) ----------
DEFAULT_CSV_PATH = r"/mnt/data/gmo_sentiment_500 (1).csv"   # <-- your CSV (from our history)
DEFAULT_MODEL_PATH = r"/mnt/data/sentiment_model.joblib"    # optional default model path
DEFAULT_VECT_PATH = r"/mnt/data/tfidf_vectorizer.joblib"    # optional default vectorizer path

# ---------- Utilities ----------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

@st.cache_data
def train_simple_model(X_train, y_train):
    vect = TfidfVectorizer(max_features=5000)
    Xtr = vect.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, y_train)
    return vect, clf

def evaluate(vect, clf, X, y):
    Xt = vect.transform(X)
    preds = clf.predict(Xt)
    acc = accuracy_score(y, preds)
    report = classification_report(y, preds, zero_division=0)
    cm = confusion_matrix(y, preds, labels=sorted(list(set(y))))
    return acc, report, cm, preds

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Sentiment Analysis (Streamlit)", layout="wide")
st.title("Sentiment Analysis — Streamlit App")
st.write("Load dataset, load / train model, predict and download artifacts.")

st.sidebar.header("Data & Models")

# --- CSV load ---
uploaded_csv = st.sidebar.file_uploader("Upload CSV (optional) — leave empty to use default", type=["csv"])
if uploaded_csv is not None:
    df = pd.read_csv(uploaded_csv)
    st.sidebar.success("CSV uploaded")
else:
    if Path(DEFAULT_CSV_PATH).exists():
        st.sidebar.info(f"Using default CSV: `{DEFAULT_CSV_PATH}`")
        df = load_csv(DEFAULT_CSV_PATH)
    else:
        st.sidebar.warning("No CSV found at default path. Please upload a CSV to proceed.")
        st.stop()

# Show dataframe preview
st.subheader("Dataset preview")
st.dataframe(df.head(10))

# Let user choose text and label columns
col_options = df.columns.tolist()
default_text_col = None
for c in col_options:
    if c.lower() in ("text", "tweet", "review", "comment", "message"):
        default_text_col = c
        break
if default_text_col is None:
    default_text_col = col_options[0]

text_col = st.sidebar.selectbox("Text column", options=col_options, index=col_options.index(default_text_col))
label_col = st.sidebar.selectbox("Label column (if exists)", options=[None] + col_options, index=0)

# Preprocess
do_clean = st.sidebar.checkbox("Clean text (lowercase, remove punctuation/urls)", value=True)
df["clean_text"] = df[text_col].astype(str)
if do_clean:
    df["clean_text"] = df["clean_text"].apply(clean_text)

st.write("### Cleaned text sample")
st.dataframe(df[["clean_text"]].head(8))

# --- Model / Vectorizer load (or upload) ---
st.sidebar.subheader("Load model & vectorizer (optional)")
model_file = None
vect_file = None

# Try default paths first
if Path(DEFAULT_MODEL_PATH).exists():
    st.sidebar.write(f"Found default model: `{DEFAULT_MODEL_PATH}`")
    try:
        model_file = joblib.load(DEFAULT_MODEL_PATH)
    except Exception:
        model_file = None

if Path(DEFAULT_VECT_PATH).exists():
    st.sidebar.write(f"Found default vectorizer: `{DEFAULT_VECT_PATH}`")
    try:
        vect_file = joblib.load(DEFAULT_VECT_PATH)
    except Exception:
        vect_file = None

# File upload fallback
uploaded_model = st.sidebar.file_uploader("Upload model (.joblib) (optional)", type=["joblib"])
if uploaded_model is not None:
    try:
        model_file = joblib.load(uploaded_model)
        st.sidebar.success("Model uploaded and loaded.")
    except Exception as e:
        st.sidebar.error(f"Failed to load uploaded model: {e}")

uploaded_vect = st.sidebar.file_uploader("Upload vectorizer (.joblib) (optional)", type=["joblib"])
if uploaded_vect is not None:
    try:
        vect_file = joblib.load(uploaded_vect)
        st.sidebar.success("Vectorizer uploaded and loaded.")
    except Exception as e:
        st.sidebar.error(f"Failed to load uploaded vectorizer: {e}")

# Determine label source
labels_available = False
if label_col:
    non_empty = df[label_col].dropna().astype(str).str.strip().ne("").sum()
    if non_empty > 0:
        labels_available = True

auto_label_opt = st.sidebar.checkbox("Auto-label missing labels using simple rules (positive/neutral/negative)", value=not labels_available)

if labels_available:
    y_series = df[label_col].astype(str)
else:
    # If no label column OR mostly empty, require auto-label or training from scratch
    if auto_label_opt:
        # Simple rule-based auto-label using presence of words (fast) — but we will train only if user asks
        def simple_label(text):
            t = text.lower()
            pos_words = ["good","great","positive","love","excellent","amazing","benefit","safe"]
            neg_words = ["bad","danger","harm","risk","concern","negative","unsafe","problem"]
            p = any(w in t for w in pos_words)
            n = any(w in t for w in neg_words)
            if p and not n:
                return "positive"
            if n and not p:
                return "negative"
            # fallback neutral
            return "neutral"
        df["auto_sentiment"] = df["clean_text"].apply(simple_label)
        y_series = df["auto_sentiment"]
        st.sidebar.info("Auto-labeling applied (simple rule-based). You can retrain the model using these labels.")
    else:
        st.warning("No labels available. Enable auto-labeling or upload a labeled CSV.")
        st.stop()

# If both vect & model present, allow predict; else allow train
if (vect_file is not None) and (model_file is not None):
    st.success("Model and vectorizer loaded — ready to predict")
    # Prediction UI
    st.subheader("Make prediction with loaded model")
    input_text = st.text_area("Enter text to predict sentiment", height=120)
    if st.button("Predict"):
        txt = clean_text(input_text) if do_clean else input_text
        Xv = vect_file.transform([txt])
        pred = model_file.predict(Xv)[0]
        st.info(f"Predicted sentiment: **{pred}**")
        try:
            probs = model_file.predict_proba(Xv)
            st.write(pd.DataFrame(probs, columns=model_file.classes_).T)
        except Exception:
            pass

    # Show evaluation if labels exist
    if "clean_text" in df.columns:
        if y_series is not None and len(set(y_series))>1:
            st.subheader("Evaluate on dataset (optional)")
            if st.button("Evaluate loaded model on dataset"):
                X = df["clean_text"].fillna("")
                y = y_series.fillna("")
                acc, report, cm, preds = evaluate(vect_file, model_file, X, y)
                st.write(f"Accuracy: **{acc:.3f}**")
                st.text(report)
                st.write("Confusion matrix:")
                st.write(cm)
else:
    st.info("Model or vectorizer missing — you can train a new model from the dataset below.")
    st.subheader("Train a new model (TF-IDF + Logistic Regression)")
    test_size = st.sidebar.slider("Test size (%)", 5, 50, 20)
    random_state = st.sidebar.number_input("Random seed", value=42, min_value=0, step=1)
    if st.button("Train model now"):
        X = df["clean_text"].fillna("")
        y = y_series.fillna("")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(test_size/100.0), random_state=int(random_state), stratify=y if len(set(y))>1 else None)
        vect, clf = train_simple_model(X_train, y_train)
        acc_train, rep_train, cm_train, _ = evaluate(vect, clf, X_train, y_train)
        acc_test, rep_test, cm_test, preds_test = evaluate(vect, clf, X_test, y_test)
        st.success(f"Training done — Test accuracy: {acc_test:.3f}")
        st.subheader("Test classification report")
        st.text(rep_test)

        # Save artifacts to disk (so you can download later)
        model_save_path = "sentiment_model.joblib"
        vect_save_path = "tfidf_vectorizer.joblib"
        joblib.dump(clf, model_save_path)
        joblib.dump(vect, vect_save_path)
        st.sidebar.download_button("Download trained model (.joblib)", data=open(model_save_path, "rb"), file_name=model_save_path)
        st.sidebar.download_button("Download vectorizer (.joblib)", data=open(vect_save_path, "rb"), file_name=vect_save_path)

        # Also offer cleaned labeled CSV
        out_csv = "sentiment_cleaned_labeled.csv"
        out_df = df.copy()
        out_df["used_label"] = y_series
        out_df.to_csv(out_csv, index=False)
        st.sidebar.download_button("Download cleaned labeled CSV", data=open(out_csv, "rb"), file_name=out_csv)

        # show some sample predictions from test
        st.subheader("Sample test predictions")
        sample_idx = list(range(min(10, len(X_test))))
        sample_texts = X_test.iloc[sample_idx]
        sample_preds = clf.predict(vect.transform(sample_texts))
        sample_df = pd.DataFrame({"text": sample_texts.values, "actual": y_test.iloc[sample_idx].values, "predicted": sample_preds})
        st.dataframe(sample_df)

st.write("---")
st.write("Notes: Default CSV path used (from this session):")
st.code(DEFAULT_CSV_PATH)
st.write("If this path does not exist in your environment, upload the CSV in the sidebar.")