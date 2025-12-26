{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b3ce5960-de5c-4a81-9441-8e1173b6f07c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import re\n",
    "import nltk\n",
    "from nltk.corpus import stopwords\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "d6469f2e-bd2e-49b2-9964-1d6c56ada3af",
   "metadata": {},
   "outputs": [],
   "source": [
    "#load the saved model and vectorizer\n",
    "svm_model= joblib.load(\"svm_model.pkl\")\n",
    "vectorizer= joblib.load(\"tfidf_vectorizer.pkl\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "8e3b5e32-6d3c-4b7c-994c-410d98ba2acf",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to C:\\Users\\Sanjana\n",
      "[nltk_data]     Uprety\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n"
     ]
    }
   ],
   "source": [
    "# Download stopwords\n",
    "nltk.download('stopwords')\n",
    "stop_words = set(stopwords.words('english'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "8ca16240-f3b5-4c62-b621-99af532fbd6c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-12-26 18:09:13.344 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-26 18:09:13.346 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-26 18:09:13.347 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-26 18:09:13.349 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-26 18:09:13.351 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-26 18:09:13.354 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-26 18:09:13.357 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-26 18:09:13.359 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-26 18:09:13.360 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-26 18:09:13.362 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-26 18:09:13.365 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-26 18:09:13.371 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-26 18:09:13.374 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-26 18:09:13.377 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-26 18:09:13.379 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-26 18:09:13.390 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-26 18:09:13.396 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-26 18:09:13.405 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-26 18:09:13.410 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "# Function to clean text\n",
    "def clean_text(text):\n",
    "    text = re.sub(r\"http\\S+|www\\S+|https\\S+\", '', text)\n",
    "    text = re.sub(r'\\@\\w+|\\#', '', text)\n",
    "    text = re.sub(r'[^\\w\\s]', '', text)\n",
    "    text = text.lower()\n",
    "    text = ' '.join([word for word in text.split() if word not in stop_words])\n",
    "    return text\n",
    "# Streamlit App\n",
    "st.title(\"Twitter Sentiment Analysis\")\n",
    "st.write(\"Enter a tweet to classify it as Hate Speech or Not Hate Speech.\")\n",
    "user_input = st.text_area(\"Enter Tweet:\", \"\")\n",
    "if st.button(\"Predict\"):\n",
    "    if user_input.strip() != \"\":\n",
    "        cleaned_text = clean_text(user_input)\n",
    "        transformed_text = vectorizer.transform([cleaned_text])\n",
    "        prediction = svm_model.predict(transformed_text)[0]\n",
    "\n",
    "        result = \"Hate Speech 😡\" if prediction == 1 else \"Not Hate Speech 😊\"\n",
    "        st.title(f\"Prediction: **{result}**\")\n",
    "    else:\n",
    "        st.title(\"Please enter a tweet.\")    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b5e1cc0e-56bc-4af9-aa24-b2922ee650ad",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0a42621f-9637-4066-8b9e-a0ab3d084017",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c410e288-ce9c-4364-9558-745f87bc33ce",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
