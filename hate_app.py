import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# MUST be the first Streamlit command
st.set_page_config(page_title="Hate Speech Detection", layout="centered")

# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# -----------------------------
# Load trained pipeline
# -----------------------------
model = joblib.load("hate_speech_pipeline.pkl")

# -----------------------------
# Text cleaning function
# (must match training)
# -----------------------------
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🚨 Hate Speech Detection App")
st.write("Enter a tweet or sentence to check whether it contains hate speech.")

user_input = st.text_area("✍️ Enter text here:", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_text = clean_text(user_input)

        # Predict directly (NO vectorizer needed)
        prediction = model.predict([cleaned_text])[0]

        if prediction == 1:
            st.error("⚠️ Hate Speech Detected")
        else:
            st.success("✅ No Hate Speech Detected")
