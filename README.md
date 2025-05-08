📰 Fake News Detection

🚀 Features
Streamlit-based user interface

Logistic Regression model for binary classification

Text preprocessing using NLTK

TF-IDF vectorization of input text


 Project Structure
 
├── app.py                  # Main application script (Streamlit)
├── fake_news_model.pkl    # Trained ML model (not included in repo)
├── tfidf_vectorizer.pkl   # TF-IDF vectorizer (not included in repo)
├── fakenews detection.ipynb # Jupyter notebook for model development


⚙️ Setup Instructions
Clone the repository
cd fake-news-detector

Install the dependencies
pip install -r requirements.txt

Download NLTK stopwords
import nltk
nltk.download('stopwords')

Run the Streamlit app
streamlit run app.py
