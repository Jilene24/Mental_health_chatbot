import pandas as pd
import nltk
import spacy
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import contractions
import string

nltk.download('punkt')
nltk.download('stopwords')

# Load necessary tools
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    if pd.isnull(text):
        return ""  # Handle missing values

    # Expand contractions
    text = contractions.fix(text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Lowercase
    tokens = [word.lower() for word in tokens]
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    # Correct spelling
    tokens = [str(TextBlob(word).correct()) for word in tokens]
    lemmatized_tokens = [token.lemma_ for token in nlp(" ".join(tokens))]
    return " ".join(lemmatized_tokens)
