import streamlit as st
import joblib
import numpy as np
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

# Download required NLTK data
nltk.download('stopwords')

# Load models and vectorizer
@st.cache_resource
def load_models():
    models = {
        'logistic_regression': joblib.load('models/logistic_regression_model.joblib'),
        'random_forest': joblib.load('models/random_forest_model.joblib'),
        'svm': joblib.load('models/svm_model.joblib')
    }
    vectorizer = joblib.load('models/vectorizer.joblib')
    return models, vectorizer

models, vectorizer = load_models()

# Preprocessing function
port_stem = PorterStemmer()
def preprocess_text(text):
    stemmed_content = re.sub('[^a-zA-Z]',' ',text)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Sentiment analysis function
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    return polarity, subjectivity

# Ensemble prediction function
def ensemble_predict(text):
    processed_text = preprocess_text(text)
    X = vectorizer.transform([processed_text])
    predictions = []
    probabilities = []
    for model in models.values():
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]
        predictions.append(pred)
        probabilities.append(prob)
    final_prediction = np.argmax(np.bincount(predictions))
    avg_probability = np.mean(probabilities, axis=0)
    return final_prediction, avg_probability

# Streamlit UI
st.title("📰 Fake News Detection System")
st.markdown("""
Type or paste a news article below and click **Analyze** to check if it's real or fake. The system uses advanced machine learning and sentiment analysis for accurate results.
""")

news_text = st.text_area("Enter news article:", height=200)

if st.button("Analyze"):
    if not news_text.strip():
        st.warning("Please enter some news text to analyze.")
    else:
        with st.spinner('Analyzing...'):
            prediction, probability = ensemble_predict(news_text)
            polarity, subjectivity = get_sentiment(news_text)

        # Display prediction with color
        if prediction == 1:
            st.markdown('<h2 style="color:red;">🚩 Fake News</h2>', unsafe_allow_html=True)
        else:
            st.markdown('<h2 style="color:green;">✅ Real News</h2>', unsafe_allow_html=True)

        # Show confidence scores
        st.subheader("Confidence Scores")
        st.progress(float(probability[1]), text=f"Fake: {probability[1]*100:.2f}%")
        st.progress(float(probability[0]), text=f"Real: {probability[0]*100:.2f}%")

        # Sentiment analysis visualization
        st.subheader("Sentiment Analysis")
        st.write(f"**Polarity:** {polarity:.2f}  ", unsafe_allow_html=True)
        st.write(f"**Subjectivity:** {subjectivity:.2f}  ", unsafe_allow_html=True)
        st.markdown("<small>Polarity: -1 (negative) to 1 (positive)<br>Subjectivity: 0 (objective) to 1 (subjective)</small>", unsafe_allow_html=True)
        st.slider("Polarity", min_value=-1.0, max_value=1.0, value=float(polarity), step=0.01, disabled=True)
        st.slider("Subjectivity", min_value=0.0, max_value=1.0, value=float(subjectivity), step=0.01, disabled=True)

        # Professional appearance
        st.markdown("---")
        st.info("This tool uses an ensemble of machine learning models and sentiment analysis to provide robust fake news detection. Suitable for journalists, researchers, and the general public.")

else:
    st.info("Enter a news article and click Analyze to get started.") 