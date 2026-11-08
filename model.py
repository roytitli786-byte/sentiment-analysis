import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 1. Download required NLP assets
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Cleans raw text by removing punctuation, numbers, and stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower() # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove punctuation & numbers
    text = ' '.join([word for word in text.split() if word not in stop_words]) # Remove stopwords
    return text

print("--- Step 1: Loading Data ---")
# Load your dataset (Adjust 'Review' and 'Rating' column names if yours are different)
df = pd.read_csv('flipkart_reviews.csv')

# Drop missing values
df = df.dropna(subset=['review', 'rating'])

print("--- Step 2: Preprocessing Text ---")
# Apply the cleaning function to the review column
df['Cleaned_Review'] = df['review'].apply(clean_text)

# Convert ratings into Binary Sentiment: 1 (Positive for 4-5 stars) and 0 (Negative for 1-3 stars)
df['Label'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)

print("--- Step 3: Splitting Dataset ---")
X = df['Cleaned_Review']
y = df['Label']

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Step 4: Converting Text to Numbers (TF-IDF) ---")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("--- Step 5: Training Logistic Regression Model ---")
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

print("--- Step 6: Evaluating the Model ---")
predictions = model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
print("\nClassification Report:\n", classification_report(y_test, predictions))

print("--- Step 7: Saving Artifacts for Deployment ---")
# Save the trained model and vectorizer to files
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Done! Artifacts saved successfully.")
