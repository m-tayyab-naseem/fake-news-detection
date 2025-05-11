import pytest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def ensemble_predict(models, X):
    """Make ensemble predictions using multiple models."""
    predictions = []
    for model in models.values():
        pred = model.predict(X)
        predictions.append(pred)
    
    # Take majority vote
    predictions_array = np.array(predictions, dtype=int)
    ensemble_pred = np.apply_along_axis(
        lambda x: np.argmax(np.bincount(x)), 
        axis=0, 
        arr=predictions_array
    )
    return ensemble_pred

def test_ensemble_predict():
    # Create sample data
    X = np.array([[1, 2], [3, 4], [5, 6]])
    
    # Create dummy models that always predict the same class
    class DummyModel:
        def __init__(self, prediction):
            self.prediction = prediction
        def predict(self, X):
            return np.array([self.prediction] * len(X), dtype=int)
    
    # Test case 1: All models agree
    models = {
        'model1': DummyModel(0),
        'model2': DummyModel(0),
        'model3': DummyModel(0)
    }
    predictions = ensemble_predict(models, X)
    assert all(pred == 0 for pred in predictions)

    # Test case 2: Majority vote (2 vs 1)
    models = {
        'model1': DummyModel(0),
        'model2': DummyModel(0),
        'model3': DummyModel(1)
    }
    predictions = ensemble_predict(models, X)
    assert all(pred == 0 for pred in predictions)

    # Test case 3: Single model
    models = {
        'model1': DummyModel(1)
    }
    predictions = ensemble_predict(models, X)
    assert all(pred == 1 for pred in predictions)

    # Test case 4: Empty models dictionary
    models = {}
    with pytest.raises(ValueError):
        ensemble_predict(models, X)

def test_model_probabilities():
    # Create a simple dataset
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    
    # Train a simple model
    model = LogisticRegression()
    model.fit(X, y)
    
    # Test probability predictions
    probs = model.predict_proba(X)
    assert probs.shape == (3, 2)  # 3 samples, 2 classes
    assert all(abs(sum(prob) - 1.0) < 1e-10 for prob in probs)  # Probabilities sum to 1
    assert all(0 <= prob <= 1 for prob in probs.flatten())  # All probabilities between 0 and 1 