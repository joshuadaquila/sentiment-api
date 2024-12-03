from flask import Flask, request, jsonify
import joblib
import re
import nltk
nltk.data.path.append('./nltk_data')
from nltk.corpus import stopwords

app = Flask(__name__)

# Load the model and vectorizer
loaded_model = joblib.load('sentiment_model.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Sentiment mapping
sentiment_mapping = {0: 'positive', 1: 'negative', 2: 'neutral'}

# API route for sentiment prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    texts = data.get('texts', [])
    texts_processed = [preprocess_text(text) for text in texts]
    texts_vectorized = loaded_vectorizer.transform(texts_processed)
    predictions = loaded_model.predict(texts_vectorized)
    sentiments = [sentiment_mapping.get(pred, 'unknown') for pred in predictions]
    return jsonify({'sentiments': sentiments})

if __name__ == '__main__':
    app.run(debug=True)
