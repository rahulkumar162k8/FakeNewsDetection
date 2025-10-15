<<<<<<< HEAD
# utils.py
import re
import joblib
import pickle
import numpy as np

# Ensure you have run: python -m nltk.downloader stopwords wordnet
# or uncomment the lines below in model_trainer.py to download them.
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def text_cleaning(text):
    """
    Cleans the input text: lowercases, removes non-alphabetic characters,
    removes stop words, and applies lemmatization. (Refactored from notebook)
    """
    if not isinstance(text, str):
        return ""
        
    # Lowercase and remove non-alphabetic characters
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces

    # Tokenize, remove stopwords, and lemmatize
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def predict_classical(model, vectorizer, text_input):
    """Predicts using classical ML models (LR, SVM, NB, RF)."""
    # 1. Clean the text using the TF-IDF cleaning logic
    cleaned_text = text_cleaning(text_input)
    # 2. Transform the text
    vectorized_text = vectorizer.transform([cleaned_text])
    # 3. Predict
    prediction = model.predict(vectorized_text)[0]
    return "True News" if prediction == 1 else "Fake News"

def predict_lstm(model, tokenizer, max_len, text_input):
    """Predicts using the LSTM model."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    # 1. Clean the text
    cleaned_text = text_cleaning(text_input)
    # 2. Tokenize and sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    # 3. Pad the sequence
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    # 4. Predict
    # The LSTM model outputs a probability (0 to 1) for class 1 (True News)
    probability = model.predict(padded_sequence)[0][0]
    prediction = 1 if probability >= 0.5 else 0

=======
# utils.py
import re
import joblib
import pickle
import numpy as np

# Ensure you have run: python -m nltk.downloader stopwords wordnet
# or uncomment the lines below in model_trainer.py to download them.
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def text_cleaning(text):
    """
    Cleans the input text: lowercases, removes non-alphabetic characters,
    removes stop words, and applies lemmatization. (Refactored from notebook)
    """
    if not isinstance(text, str):
        return ""
        
    # Lowercase and remove non-alphabetic characters
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces

    # Tokenize, remove stopwords, and lemmatize
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def predict_classical(model, vectorizer, text_input):
    """Predicts using classical ML models (LR, SVM, NB, RF)."""
    # 1. Clean the text using the TF-IDF cleaning logic
    cleaned_text = text_cleaning(text_input)
    # 2. Transform the text
    vectorized_text = vectorizer.transform([cleaned_text])
    # 3. Predict
    prediction = model.predict(vectorized_text)[0]
    return "True News" if prediction == 1 else "Fake News"

def predict_lstm(model, tokenizer, max_len, text_input):
    """Predicts using the LSTM model."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    # 1. Clean the text
    cleaned_text = text_cleaning(text_input)
    # 2. Tokenize and sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    # 3. Pad the sequence
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    # 4. Predict
    # The LSTM model outputs a probability (0 to 1) for class 1 (True News)
    probability = model.predict(padded_sequence)[0][0]
    prediction = 1 if probability >= 0.5 else 0

>>>>>>> 7d4d2e3e66fa11c460251441040bdb763edfdc55
    return "True News" if prediction == 1 else "Fake News", probability