import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Setup NLP stop words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load the saved machine learning artifacts
@st.cache_resource
def load_artifacts():
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Please run model.py first!")
    st.stop()

# Build the Web Application Layout
st.title("🛍️ Flipkart Review Sentiment Analyzer")
st.write("Type or paste any product review below to see how our AI model classifies it.")

user_review = st.text_area("Product Review Input", placeholder="Type your product review here...", height=150)

if st.button("Analyze Sentiment"):
    if user_review.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Preprocess and vectorize the user input text
        cleaned = clean_text(user_review)
        vectorized = vectorizer.transform([cleaned])
        
        # Make predictions using your trained model
        prediction = model.predict(vectorized)
        probability = model.predict_proba(vectorized)
        confidence = max(probability[0]) * 100

        # Render outputs beautifully
        st.write("---")
        if prediction[0] == 1:
            st.success("**Result:** Positive Review 👍")
        else:
            st.error("**Result:** Negative Review 👎")
            
        st.info(f"**Confidence Score:** {confidence:.2f}%")

        