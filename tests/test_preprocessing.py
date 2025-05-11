import pytest
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download required NLTK data
nltk.download('stopwords')

def preprocess_text(text):
    """Preprocess the input text using the same method as training."""
    port_stem = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]',' ',text)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

def test_preprocess_text():
    # Test case 1: Basic text preprocessing
    text = "Hello, World! This is a test."
    processed = preprocess_text(text)
    assert isinstance(processed, str)
    assert processed.islower()
    assert not any(char.isupper() for char in processed)
    assert not any(char in ".,!?" for char in processed)

    # Test case 2: Text with numbers and special characters
    text = "Test123!@# Hello456"
    processed = preprocess_text(text)
    assert not any(char.isdigit() for char in processed)
    assert not any(char in "!@#" for char in processed)

    # Test case 3: Empty string
    text = ""
    processed = preprocess_text(text)
    assert processed == ""

    # Test case 4: Text with stopwords
    text = "the quick brown fox jumps over the lazy dog"
    processed = preprocess_text(text)
    assert "the" not in processed.split()
    assert "over" not in processed.split()

    # Test case 5: Text with stemming
    text = "running runs ran"
    processed = preprocess_text(text)
    words = processed.split()
    # Check that all words are stemmed to their root form
    assert all(word in ['run', 'ran'] for word in words) 