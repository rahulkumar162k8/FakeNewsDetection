# model_trainer.py
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import nltk
from utils import text_cleaning # Import the cleaning function

# Scikit-learn models
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Deep learning models
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Configuration (Based on typical notebook parameters) ---
RANDOM_SEED = 42
MAX_WORDS = 10000 
MAX_LEN = 300     
EMBEDDING_DIM = 100 
OUTPUT_DIR = 'models'

def load_data():
    """Load, combine, and clean the data based on the notebook's initial steps."""
    print("Loading and preparing data...")
    try:
        # Load datasets
        fake_df = pd.read_csv('Fake.csv', low_memory=False)
        true_df = pd.read_csv('True.csv', low_memory=False)
    except FileNotFoundError:
        print("--- ERROR ---")
        print("Please place 'Fake.csv' and 'True.csv' in the same directory as this script.")
        print("-----------------")
        return None

    # Label, Concatenate, and Preprocess Columns
    fake_df['label'] = 0
    true_df['label'] = 1
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df.drop_duplicates(inplace=True)
    df = df.drop(['subject', 'date'], axis=1)
    df['total'] = df['title'] + ' ' + df['text'] # Combine title and text
    df.drop(['title', 'text'], axis=1, inplace=True)
    
    # Apply Cleaning
    df['cleaned_text'] = df['total'].apply(text_cleaning)

    X = df['cleaned_text']
    y = df['label']
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

def train_classical_models(X_train_text, X_test_text, y_train, y_test):
    """Train and save TF-IDF vectorizer and classical ML models."""
    print("\nTraining Classical Models (TF-IDF)...")

    # Train TF-IDF Vectorizer
    # Max_features set to 5000 as a reasonable assumption for the notebook's TfidfVectorizer.
    vectorizer = TfidfVectorizer(max_features=5000) 
    X_train_vec = vectorizer.fit_transform(X_train_text)
    
    # Save Vectorizer
    joblib.dump(vectorizer, os.path.join(OUTPUT_DIR, 'tfidf_vectorizer.pkl'))
    print("-> TF-IDF Vectorizer saved.")

    models_to_train = {
        'LogisticRegression': LogisticRegression(random_state=RANDOM_SEED, solver='liblinear'),
        'SVM': SVC(kernel='linear', random_state=RANDOM_SEED, probability=True),
        'MultinomialNB': MultinomialNB(),
        'RandomForest': RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=100)
    }

    for name, model in models_to_train.items():
        print(f"  -> Training {name}...")
        model.fit(X_train_vec, y_train)
        joblib.dump(model, os.path.join(OUTPUT_DIR, f'{name.lower()}_model.pkl'))
        print(f"  -> {name} model saved.")

def train_lstm_model(X_train_text, X_test_text, y_train, y_test):
    """Train and save Keras Tokenizer and LSTM model."""
    print("\nTraining LSTM Model (Keras)...")

    # 1. Tokenization and Padding
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<oov>")
    tokenizer.fit_on_texts(X_train_text)

    # Save Tokenizer
    with open(os.path.join(OUTPUT_DIR, 'lstm_tokenizer.pkl'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("-> LSTM Tokenizer saved.")

    X_train_sequences = tokenizer.texts_to_sequences(X_train_text)
    X_train_padded = pad_sequences(X_train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    # 2. Build LSTM Model
    model = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
        LSTM(128), 
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

    # 3. Train Model
    # Note: Epochs reduced to 3 for faster local training/testing. 
    # Use higher epochs (e.g., 10) for better final performance.
    model.fit(
        X_train_padded, y_train,
        epochs=3, 
        batch_size=32, 
        verbose=1
    )

    # 4. Save Model 
    model.save(os.path.join(OUTPUT_DIR, 'lstm_model.h5'))
    print("-> LSTM model saved.")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # Ensure NLTK data is available
    try:
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
    except LookupError:
        print("Downloading NLTK data (stopwords, wordnet)...")
        nltk.download('stopwords')
        nltk.download('wordnet')

    df_split = load_data()
    if df_split is not None:
        X_train_text, X_test_text, y_train, y_test = df_split

        train_classical_models(X_train_text, X_test_text, y_train, y_test)
        train_lstm_model(X_train_text, X_test_text, y_train, y_test)

        print("\n==============================================")
        print("SUCCESS: All assets are saved in the 'models' directory.")
        print("You can now proceed to deploy the folder to Streamlit.io.")
        print("==============================================")