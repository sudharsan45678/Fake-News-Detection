import streamlit as st
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import joblib

# Load the saved model and vectorizer
model = joblib.load(r'C:\Users\HP\fake_news_model.pkl')
tfidf = joblib.load(r'C:\Users\HP\tfidf_vectorizer.pkl')

# Ensure necessary NLTK resources are available
nltk.download('stopwords')

# Text cleaning function for the user input
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit app UI
st.title("Fake News Detection")
st.write("""
This is a simple fake news detection app built using a logistic regression model. 
You can input a news article, and the app will predict whether it is fake or real.
""")

# Text input
user_input = st.text_area("Enter News Title:", "")

# Prediction
if st.button("Predict"):
    if user_input:
        # Clean the user input
        cleaned_input = clean_text(user_input)
        
        # Vectorize the input
        input_vec = tfidf.transform([cleaned_input])
        
        # Predict the probability of the news being fake
        prediction_proba = model.predict_proba(input_vec)[:, 1]  # Probability of class 1 (fake news)
        
        # Output prediction
        if prediction_proba > 0.5:
            st.write(f"Prediction: Fake News (Probability: {prediction_proba[0]:.2f})")
        else:
            st.write(f"Prediction: Real News (Probability: {1 - prediction_proba[0]:.2f})")
    else:
        st.write("Please enter some text to predict.")
