import pytest
from textblob import TextBlob

def get_sentiment(text):
    """Get sentiment scores for a text."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    return polarity, subjectivity

def test_get_sentiment():
    # Test case 1: Positive text
    text = "I love this amazing product! It's wonderful!"
    polarity, subjectivity = get_sentiment(text)
    assert polarity > 0
    assert 0 <= subjectivity <= 1

    # Test case 2: Negative text
    text = "This is terrible. I hate it."
    polarity, subjectivity = get_sentiment(text)
    assert polarity < 0
    assert 0 <= subjectivity <= 1

    # Test case 3: Neutral text
    text = "The sky is blue."
    polarity, subjectivity = get_sentiment(text)
    assert abs(polarity) < 0.5
    assert 0 <= subjectivity <= 1

    # Test case 4: Empty string
    text = ""
    polarity, subjectivity = get_sentiment(text)
    assert polarity == 0
    assert subjectivity == 0

    # Test case 5: Very subjective text
    text = "I think, in my opinion, maybe this could be good."
    polarity, subjectivity = get_sentiment(text)
    assert subjectivity > 0.5

    # Test case 6: Very objective text
    text = "The temperature is 25 degrees Celsius."
    polarity, subjectivity = get_sentiment(text)
    assert subjectivity < 0.5 