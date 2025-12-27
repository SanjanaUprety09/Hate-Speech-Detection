import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (only first time)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -----------------------------
# Load saved model & vectorizer
# -----------------------------
svm_model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# -----------------------------
# Text cleaning function
# (MUST match training code)
# -----------------------------
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Hate Speech Detection", layout="centered")

st.title("🚨 Hate Speech Detection App")
st.write("Enter a tweet or sentence to check whether it contains hate speech.")

user_input = st.text_area("✍️ Enter text here:", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess
        cleaned_text = clean_text(user_input)

        # Vectorize
        text_vector = vectorizer.transform([cleaned_text])

        # Predict
        prediction = svm_model.predict(text_vector)[0]

        # Output
        if prediction == 1:
            st.error("⚠️ Hate Speech Detected")
        else:
            st.success("✅ No Hate Speech Detected")
