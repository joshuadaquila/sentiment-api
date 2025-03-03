import nltk
import joblib
import re
import numpy as np
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from flask import Flask, request, jsonify
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Download necessary NLTK corpora
try:
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')
except Exception as e:
    print(f"Error downloading NLTK corpora: {e}")

app = Flask(__name__)

# Load the models and vectorizers
sentiment_model = joblib.load('sentiment_model.pkl')  # Logistic Regression model for sentiment analysis
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')  # TF-IDF vectorizer for sentiment model
count_vectorizer = joblib.load('count_vectorizer.pkl')  # CountVectorizer for LDA model

# Load Linear Discriminant Analysis model
lda_model = joblib.load('lda_model.pkl')  # LDA model (Linear Discriminant Analysis)

# Check if lda_model is an instance of LinearDiscriminantAnalysis
if not isinstance(lda_model, LinearDiscriminantAnalysis):
    raise TypeError("The loaded LDA model is not an instance of LinearDiscriminantAnalysis.")

# Sentiment mapping for logistic regression output (0: positive, 1: negative, 2: neutral)
sentiment_mapping = {0: 'positive', 1: 'negative', 2: 'neutral'}

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

# Function to extract adjectives from text
def extract_adjectives(text):
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    adjectives = [word for word, tag in tagged_words if tag in ['JJ', 'JJR', 'JJS']]  # Adjective tags
    return adjectives


# Function to explain sentiment predictions
def get_sentiment_explanation(text, sentiment_prediction, vectorizer, model):
    # Vectorize the text input
    text_vectorized = vectorizer.transform([text])

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Get the coefficients from the logistic regression model
    coefficients = model.coef_.flatten()

    # Number of top words to extract
    top_n = 5

    # Determine top words based on sentiment
    if sentiment_prediction == 'positive':
        top_words_indices = np.argsort(coefficients)[-top_n:][::-1]
    elif sentiment_prediction == 'negative':
        top_words_indices = np.argsort(coefficients)[:top_n]
    else:
        top_words_indices = np.argsort(coefficients)[len(coefficients) // 2:len(coefficients) // 2 + top_n]

    # Ensure indices are within bounds
    top_words_indices = [i for i in top_words_indices if i < len(feature_names)]

    # Extract corresponding words and their coefficients
    top_words = [feature_names[i] for i in top_words_indices]
    top_coefficients = coefficients[top_words_indices]

    # Extract adjectives from the text
    adjectives = extract_adjectives(text)

    # Prepare the explanation dictionary
    explanation = {
        "sentiment": sentiment_prediction,
        "top_words": top_words,
        "coefficients": top_coefficients.tolist(),
        "adjectives": adjectives
    }

    return explanation

@app.route('/predictEvent', methods=['POST'])
def predictEvent():
    data = request.json
    print("Received data:", data)  # Debugging: print the received data

    # Validate the structure of the input data
    if 'texts' not in data:
        return jsonify({"error": "'texts' field is missing in the input data"}), 400

    texts_with_eventId = data['texts']

    # Check if texts_with_eventId is a list of dictionaries
    if not isinstance(texts_with_eventId, list) or not all(isinstance(text, dict) for text in texts_with_eventId):
        return jsonify({"error": "'texts' should be a list of dictionaries"}), 400

    # Check if 'content' key exists in each text dictionary
    for text in texts_with_eventId:
        if 'content' not in text or not isinstance(text['content'], str):
            return jsonify({"error": "'content' key missing or invalid in one of the text entries"}), 400

    if not texts_with_eventId or all(text['content'].strip() == '' for text in texts_with_eventId):
        return jsonify({"error": "No feedback provided or all texts are empty."}), 400

    try:
        # Preprocess the texts
        texts_processed = [preprocess_text(text['content']) for text in texts_with_eventId]
        print(f"Processed texts: {texts_processed}")  # Check processed texts

        # Vectorize the texts using TF-IDF for sentiment prediction
        texts_vectorized_tfidf = tfidf_vectorizer.transform(texts_processed)
        print(f"Vectorized texts (TF-IDF): {texts_vectorized_tfidf.shape}")  # Check vectorized shape

        # Sentiment prediction using logistic regression model
        sentiment_predictions = sentiment_model.predict(texts_vectorized_tfidf)
        print(f"Sentiment predictions: {sentiment_predictions}")

        # Map predictions to sentiment labels
        sentiments = [sentiment_mapping.get(int(pred), 'unknown') for pred in sentiment_predictions]

        # Get explanations for the sentiment predictions from LR model
        sentiment_explanations = []
        for i, sentiment in enumerate(sentiments):
            explanation = get_sentiment_explanation(
                texts_with_eventId[i]['content'],  # The text content
                sentiment,  # Sentiment prediction (positive, negative, neutral)
                tfidf_vectorizer,  # The TF-IDF vectorizer used for the model
                sentiment_model  # The Logistic Regression model
            )
            print(f"Explanation for text {i} (Event ID: {texts_with_eventId[i]['eventId']}): {explanation}")  # Debugging: print explanations
            sentiment_explanations.append({
                'eventId': texts_with_eventId[i]['eventId'],  # Include event Id with explanation
                'explanation': explanation  # The explanation from the LR model
            })

        # Get the most mentioned words from the texts for LR (general important words)
        top_words_lr = get_most_mentioned_words_for_sentiment(texts_processed, tfidf_vectorizer)
        print(f"Top words for LR: {top_words_lr}")

        # Get the most mentioned words related to the sentiment (adjectives)
        sentiment_related_words = {}
        for sentiment in sentiment_mapping.values():
            sentiment_related_words[sentiment] = get_sentiment_related_words_for_sentiment(texts_processed, sentiment)
        print(f"Sentiment related words: {sentiment_related_words}")

        # Vectorize the texts using the CountVectorizer for LDA topic prediction
        texts_vectorized_count = count_vectorizer.transform(texts_processed)

        # Linear Discriminant Analysis (LDA) topic prediction
        lda_predictions = lda_model.predict(texts_vectorized_count.toarray())  # Use predict method for LDA
        print(f"LDA Predictions: {lda_predictions}")

        # Get LDA explanations for each text
        lda_explanations = [
            {
                'eventId': texts_with_eventId[i]['eventId'],  # Include eventId with LDA explanation
                'lda_explanation': get_lda_explanation(lda_model, count_vectorizer, text)
            }
            for i, text in enumerate(texts_processed)
        ]

        # Prepare the response with the model results
        response = {
            'lda': lda_predictions.tolist(),  # Just the raw topic numbers from LDA
            'lda_explanations': lda_explanations,  # LDA explanations for each text
            'lr': sentiments,  # Sentiment predictions from LR
            'topwords': top_words_lr,  # Top mentioned words for LR sentiment prediction
            'sentiment_related_words': sentiment_related_words,  # Adjectives related to the predicted sentiment
            'sentiment_explanations': sentiment_explanations  # Explanation of sentiment predictions from LR
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Error in prediction: {e}"}), 500

# Function to get the most mentioned words for sentiment
def get_most_mentioned_words_for_sentiment(texts, vectorizer, top_n=10):
    texts_vectorized = vectorizer.transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    summed_tfidf = np.array(texts_vectorized.sum(axis=0)).flatten()
    sorted_indices = summed_tfidf.argsort()[-top_n:][::-1]
    top_words = [feature_names[i] for i in sorted_indices]

    return top_words

# Function to get the most mentioned sentiment-related words for each sentiment
def get_sentiment_related_words_for_sentiment(texts, sentiment, top_n=10):
    sentiment_related_words = []
    for text in texts:
        adjectives = extract_adjectives(text)
        sentiment_related_words.extend(adjectives)

    word_freq = nltk.FreqDist(sentiment_related_words)
    most_common_words = word_freq.most_common(top_n)

    return [word for word, count in most_common_words]

# Function to explain LDA predictions
def get_lda_explanation(lda_model, count_vectorizer, text):
    """
    Provides an explanation of the LDA prediction based on the LDA model's parameters
    and the top words contributing to the predicted class.
    """
    # Vectorize the text for feature extraction
    text_vectorized = count_vectorizer.transform([text])

    # Get the predicted class
    predicted_class = lda_model.predict(text_vectorized)[0]

    # Get feature names (words)
    words = count_vectorizer.get_feature_names_out()

    # Get coefficients for the predicted class
    coefficients = lda_model.coef_[predicted_class]

    # Convert the vectorized text to an array
    feedback_vector = text_vectorized.toarray()[0]

    # Identify words in the feedback and their corresponding contributions
    word_contributions = zip(words, feedback_vector * coefficients)
    sorted_word_contributions = sorted(word_contributions, key=lambda x: x[1], reverse=True)

    # Extract the top 5 contributing words
    top_words = [word for word, _ in sorted_word_contributions[:5]]
    top_coefficients = [contrib for _, contrib in sorted_word_contributions[:5]]

    lda_explanation = {
        "predicted_class": int(predicted_class),  # Convert to standard Python int
        "top_words": top_words,
        "top_coefficients": top_coefficients  # Include coefficients for the top words
    }

    return lda_explanation

@app.route('/healthz', methods=['GET'])
def health_check():
    try:
        # Simple response to confirm the server is running
        return jsonify({"status": "healthy", "message": "Service is running."}), 200
    except Exception as e:
        # In case of an error, respond with an internal server error status
        return jsonify({"status": "unhealthy", "message": str(e)}), 500

# Update the predict function to include LDA explanations
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print("Received data:", data)  # Debugging: print the received data
    texts = data.get('texts', [])

    # Check if any text is empty
    if not texts or all(text.strip() == '' for text in texts):
        return jsonify({"error": "No text provided or all texts are empty."}), 400

    try:
        # Preprocess the texts
        texts_processed = [preprocess_text(text) for text in texts]
        print(f"Processed texts: {texts_processed}")  # Check processed texts

        # Vectorize the texts using TF-IDF for sentiment prediction
        texts_vectorized_tfidf = tfidf_vectorizer.transform(texts_processed)
        print(f"Vectorized texts (TF-IDF): {texts_vectorized_tfidf.shape}")  # Check vectorized shape

        # Sentiment prediction using logistic regression model
        sentiment_predictions = sentiment_model.predict(texts_vectorized_tfidf)
        print(f"Sentiment predictions: {sentiment_predictions}")

        # Map predictions to sentiment labels
        sentiments = [sentiment_mapping.get(int(pred), 'unknown') for pred in sentiment_predictions]

        # Get explanations for the sentiment predictions
        sentiment_explanations = []
        for i, sentiment in enumerate(sentiments):
            explanation = get_sentiment_explanation(texts[i], sentiment, tfidf_vectorizer, sentiment_model)
            sentiment_explanations.append(explanation)

        # Get the most mentioned words from the texts for LR (general important words)
        top_words_lr = get_most_mentioned_words_for_sentiment(texts, tfidf_vectorizer)
        print(f"Top words for LR: {top_words_lr}")

        # Get the most mentioned words related to the sentiment (adjectives)
        sentiment_related_words = {}
        for sentiment in sentiment_mapping.values():
            sentiment_related_words[sentiment] = get_sentiment_related_words_for_sentiment(texts, sentiment)
        print(f"Sentiment related words: {sentiment_related_words}")

        # Vectorize the texts using the CountVectorizer for LDA topic prediction
        texts_vectorized_count = count_vectorizer.transform(texts_processed)

        # Linear Discriminant Analysis (LDA) topic prediction
        lda_predictions = lda_model.predict(texts_vectorized_count.toarray())  # Use predict method for LDA
        print(f"LDA Predictions: {lda_predictions}")

        # Get LDA explanations for each text
        lda_explanations = [get_lda_explanation(lda_model, count_vectorizer, text) for text in texts_processed]

        # Prepare the response with the model results
        response = {
            'lda': lda_predictions.tolist(),  # Just the raw topic numbers from LDA
            'lda_explanations': lda_explanations,  # LDA explanations for each text
            'lr': sentiments,  # Sentiment predictions from LR
            'topwords': top_words_lr,  # Top mentioned words for LR sentiment prediction
            'sentiment_related_words': sentiment_related_words,  # Adjectives related to the predicted sentiment
            'sentiment_explanations': sentiment_explanations  # Explanation of sentiment predictions
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Error in prediction: {e}"}), 500





if __name__ == '__main__':
    app.run(debug=True)