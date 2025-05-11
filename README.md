# Fake News Detection System

A machine learning-based system for detecting fake news using ensemble methods and sentiment analysis.

## Overview

This project implements a fake news detection system using multiple machine learning models (Logistic Regression, Random Forest, and SVM) combined through ensemble methods. It also includes sentiment analysis to provide additional insights about the news content.

## Features

- Ensemble-based fake news detection using multiple ML models
- Sentiment analysis of news content
- Text preprocessing and feature extraction
- Model training and evaluation
- Interactive testing interface
- Comprehensive test suite

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fake-news-detection
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
fake-news-detection/
├── main.py              # Main training script
├── test_news.py         # Interactive testing interface
├── tests/               # Unit tests
│   ├── test_models.py
│   ├── test_preprocessing.py
│   └── test_sentiment.py
├── models/              # Saved model files
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Usage

### Training the Models

To train the models:
```bash
python main.py
```

This will:
- Load and preprocess the training data
- Train multiple models (Logistic Regression, Random Forest, SVM)
- Create an ensemble model
- Save the trained models to the `models/` directory

### Testing News Articles

To test news articles:
```bash
python test_news.py
```

This will start an interactive session where you can:
- Enter news articles for analysis
- Get predictions (Real/Fake)
- View confidence scores
- See sentiment analysis results

## Model Details

### Ensemble Method
The system uses three models:
1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM)

Predictions are made using majority voting from all three models.

### Sentiment Analysis
Uses TextBlob to analyze:
- Polarity (-1 to 1): Negative to positive sentiment
- Subjectivity (0 to 1): Objective to subjective content

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Dependencies

- numpy
- pandas
- scikit-learn
- nltk
- textblob
- joblib

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 