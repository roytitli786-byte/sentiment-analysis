# Flipkart Reviews Sentiment Analysis Dashboard

An end-to-end Machine Learning and Natural Language Processing (NLP) application that classifies e-commerce product reviews as positive or negative. The project features a trained classification backend achieving **88% overall accuracy** and an interactive user interface built with Streamlit.

## 🚀 Live Demo
* [Insert your live Streamlit link here after deployment]*

## ✨ Features
- **Data Cleaning Pipeline:** Preprocesses raw, unstructured text using NLTK by removing punctuation, special characters, numerical noise, and standardized stop words.
- **Feature Extraction:** Converts raw text strings into structured numerical matrices using a **TF-IDF Vectorizer** (Term Frequency-Inverse Document Frequency) restricted to the top 5,000 features.
- **Machine Learning Core:** Utilizes a **Logistic Regression Classifier** evaluated on an 80/20 train-test data split.
- **Web UI:** Provides an interactive text-area dashboard that calculates real-time sentiment classifications alongside a probabilistic **Confidence Score**.

## 📊 Performance Metrics
The baseline model achieves strong discriminative performance across standard evaluation metrics:
- **Overall Accuracy:** 88%
- **Positive Sentiment (1) F1-Score:** 0.93
- **Negative Sentiment (0) F1-Score:** 0.59

## 🛠️ Tech Stack
- **Language:** Python 3.12+
- **Machine Learning & Data:** Scikit-Learn, Pandas, NumPy
- **Natural Language Processing:** NLTK, Regular Expressions (`re`)
- **Deployment & UI:** Streamlit

## 📦 Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd sentiment-analysis
   ```
2. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn nltk streamlit
   ```
3. **Train the model:**
   Run the backend training script to generate the serialized pipeline components (`sentiment_model.pkl` and `tfidf_vectorizer.pkl`):
   ```bash
   python model.py
   ```
4. **Launch the web application:**
   ```bash
   python -m streamlit run app.py
   ```

## 💡 What I Learned
- Handled case-sensitive column schema mapping bugs while parsing real-world CSV files.
- Implemented text tokenization thresholds to manage mixed-sentiment user inputs gracefully (e.g., handling compound statements with mixed pros/cons).
- Serialized model states using Python's `pickle` utility to separate training computations from application runtime logic.
-

