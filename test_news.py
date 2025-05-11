import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from textblob import TextBlob
import numpy as np

# Download required NLTK data
nltk.download('stopwords')

def get_sentiment(text):
    """Get sentiment scores for a text."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    return polarity, subjectivity

def preprocess_text(text):
    """Preprocess the input text using the same method as training."""
    port_stem = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]',' ',text)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

def predict_news(text):
    """Predict if the given text is fake news using ensemble method."""
    # Load all models and vectorizer
    models = {
        'logistic_regression': joblib.load('models/logistic_regression_model.joblib'),
        'random_forest': joblib.load('models/random_forest_model.joblib'),
        'svm': joblib.load('models/svm_model.joblib')
    }
    vectorizer = joblib.load('models/vectorizer.joblib')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Get sentiment scores
    polarity, subjectivity = get_sentiment(text)
    
    # Transform the text using the vectorizer
    X = vectorizer.transform([processed_text])
    
    # Get predictions from each model
    predictions = []
    probabilities = []
    for name, model in models.items():
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]
        predictions.append(pred)
        probabilities.append(prob)
    
    # Take majority vote for final prediction
    final_prediction = np.argmax(np.bincount(predictions))
    
    # Average the probabilities from all models
    avg_probability = np.mean(probabilities, axis=0)
    
    return final_prediction, avg_probability, polarity, subjectivity

def main():
    print("Fake News Detection System (Ensemble Method)")
    print("-------------------------------------------")
    
    while True:
        print("\nEnter a news article (or 'quit' to exit):")
        text = input("> ")
        
        if text.lower() == 'quit':
            break
            
        if not text.strip():
            print("Please enter some text!")
            continue
            
        prediction, probability, polarity, subjectivity = predict_news(text)
        
        print("\nPrediction Results:")
        print("------------------")
        print(f"Prediction: {'Fake' if prediction == 1 else 'Real'} News")
        print(f"Confidence: {probability[prediction]*100:.2f}%")
        print(f"Real probability: {probability[0]*100:.2f}%")
        print(f"Fake probability: {probability[1]*100:.2f}%")
        
        print("\nSentiment Analysis:")
        print("------------------")
        print(f"Polarity: {polarity:.2f} (-1 to 1, where -1 is very negative and 1 is very positive)")
        print(f"Subjectivity: {subjectivity:.2f} (0 to 1, where 0 is very objective and 1 is very subjective)")

if __name__ == "__main__":
    main() 