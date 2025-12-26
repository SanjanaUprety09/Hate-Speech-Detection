import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
#load the saved model and vectorizer
svm_model= joblib.load("svm_model.pkl")
vectorizer= joblib.load("tfidf_vectorizer.pkl")
# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# Function to clean text
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text
# Streamlit App
st.title("Twitter Sentiment Analysis")
st.write("Enter a tweet to classify it as Hate Speech or Not Hate Speech.")
# Input area
user_input = st.text_area("Enter Tweet:", "")

# Prediction button
if st.button("Predict"):
    if user_input.strip():  # Checks if input is not empty
        # Preprocess and transform
        cleaned_text = clean_text(user_input)
        transformed_text = vectorizer.transform([cleaned_text])

        # Make prediction
        prediction = svm_model.predict(transformed_text)[0]

        # Display result
        result = "Hate Speech 😡" if prediction == 1 else "Not Hate Speech 😊"
        st.success(f"Prediction: **{result}**")
    else:
        st.warning("⚠️ Please enter a tweet before predicting.") 
