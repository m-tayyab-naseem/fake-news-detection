import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path
from textblob import TextBlob

import nltk
nltk.download('stopwords')

news_dataset = pd.read_csv('train.csv', on_bad_lines='skip')
news_dataset['content'] = news_dataset['subject']+' '+news_dataset['title']

# Add sentiment analysis
def get_sentiment(text):
    """Get sentiment scores for a text."""
    # Create a TextBlob object
    blob = TextBlob(text)
    # Get polarity (-1 to 1, where -1 is very negative and 1 is very positive)
    polarity = blob.sentiment.polarity
    # Get subjectivity (0 to 1, where 0 is very objective and 1 is very subjective)
    subjectivity = blob.sentiment.subjectivity
    return polarity, subjectivity

# Add sentiment scores to the dataset
print("Adding sentiment analysis...")
news_dataset['sentiment_polarity'], news_dataset['sentiment_subjectivity'] = zip(*news_dataset['content'].apply(get_sentiment))

# separating the data & label
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']

port_stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content
news_dataset['content'] = news_dataset['content'].apply(stemming)

#separating the data and label
X = news_dataset['content'].values
Y = news_dataset['label'].values

# converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)

# Initialize multiple models
models = {
    'logistic_regression': LogisticRegression(),
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'svm': SVC(probability=True, random_state=42)
}

# Train each model
trained_models = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, Y_train)
    trained_models[name] = model
    
    # Calculate and print accuracy for each model
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_accuracy = accuracy_score(train_pred, Y_train)
    test_accuracy = accuracy_score(test_pred, Y_test)
    print(f"{name} - Training accuracy: {train_accuracy:.4f}")
    print(f"{name} - Test accuracy: {test_accuracy:.4f}")

# Function to make ensemble predictions
def ensemble_predict(X):
    predictions = []
    for model in trained_models.values():
        pred = model.predict(X)
        predictions.append(pred)
    
    # Take majority vote
    ensemble_pred = np.apply_along_axis(
        lambda x: np.argmax(np.bincount(x)), 
        axis=0, 
        arr=np.array(predictions)
    )
    return ensemble_pred

# Calculate ensemble accuracy
ensemble_train_pred = ensemble_predict(X_train)
ensemble_test_pred = ensemble_predict(X_test)
ensemble_train_accuracy = accuracy_score(ensemble_train_pred, Y_train)
ensemble_test_accuracy = accuracy_score(ensemble_test_pred, Y_test)

print("\nEnsemble Results:")
print(f"Ensemble - Training accuracy: {ensemble_train_accuracy:.4f}")
print(f"Ensemble - Test accuracy: {ensemble_test_accuracy:.4f}")

# Save all models and vectorizer
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)

for name, model in trained_models.items():
    joblib.dump(model, models_dir / f'{name}_model.joblib')
joblib.dump(vectorizer, models_dir / 'vectorizer.joblib')
print("\nAll models and vectorizer saved successfully!")

# Example prediction using ensemble
X_new = X_test[3]
ensemble_prediction = ensemble_predict(X_new.reshape(1, -1))
print("\nExample prediction using ensemble:")
if (ensemble_prediction[0]==0):
    print('The news is Real')
else:
    print('The news is Fake')
print(f"Actual label: {Y_test[3]}")