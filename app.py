# app.py
import streamlit as st
import os
import joblib
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# Local utility functions
from utils import predict_classical, predict_lstm

import nltk 
# Make sure 'nltk' is imported


# --- App Configuration ---
st.set_page_config(page_title="AIML Fake News Detector", layout="wide")

# --- Model Loading (Caching to speed up app) ---
# Use st.cache_resource for heavy assets like Keras models
@st.cache_resource

def download_nltk_data():
    import nltk 
    
    # List of required NLTK resources
    required_resources = ['stopwords', 'wordnet'] 

    for resource in required_resources:
        try:
            # Check if resource exists; raises LookupError if not found
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            # Download only if not found
            nltk.download(resource, quiet=True) 

# Execute the download check
download_nltk_data()

def load_assets():
    """Loads all models and preprocessors from the 'models' directory."""
    models = {}
    preprocessors = {}
    model_dir = 'models'

    try:
        # Load TF-IDF Vectorizer 
        preprocessors['tfidf_vectorizer'] = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
        
        # Load Classical Models (Logistic Regression, SVM, Naive Bayes, Random Forest)
        classical_models = ['LogisticRegression', 'SVM', 'MultinomialNB', 'RandomForest']
        for name in classical_models:
            models[name] = joblib.load(os.path.join(model_dir, f'{name.lower()}_model.pkl'))

        # Load LSTM Model and Tokenizer
        models['LSTM'] = load_model(os.path.join(model_dir, 'lstm_model.h5'))
        with open(os.path.join(model_dir, 'lstm_tokenizer.pkl'), 'rb') as handle:
            preprocessors['lstm_tokenizer'] = pickle.load(handle)
        
        # LSTM specific parameters (must be consistent with training)
        preprocessors['lstm_max_len'] = 300 
            
    except Exception as e:
        # Display an error if files are missing, guiding the user to run the trainer script
        st.error("--- Deployment Error ---")
        st.error("Required model files are not found. Did you run 'model_trainer.py' locally?")
        st.error(f"Please check that the 'models' directory exists and contains all required .pkl and .h5 files. Error details: {e}")
        st.stop()
        
    return models, preprocessors

# --- Main Streamlit Application ---

st.title("📰 AI/ML Multi-Model Fake News Detector")
st.markdown("Select a model and enter a news article's text to check if it's Fake or True.")

# Load models and preprocessors
models, preprocessors = load_assets()

# Sidebar for Model Selection and Info
model_names = list(models.keys())
selected_model = st.sidebar.selectbox(
    "Select Model for Prediction", 
    model_names, 
    key='model_selector_key_2'  # <-- ADD THIS UNIQUE KEY
)

st.sidebar.markdown("---")
st.sidebar.markdown("**About the Models**")
st.sidebar.markdown(f"**Best Model (from training analysis):** LSTM")
st.sidebar.markdown("""
* **LSTM:** Recurrent Neural Network (Deep Learning) - Tokenizer/Embedding based.
* **Classical Models:** Trained using TF-IDF (Term Frequency-Inverse Document Frequency) features.
""")


# Text Input Area
text_input = st.text_area(
    "Paste the News Article Title and Body Text Here:",
    "Enter the title followed by the full text of a news article (as the models were trained on the combined text).",
    height=300,
    key='news_article_input_2' # <-- ADD THIS UNIQUE KEY
)

# Prediction Button
if st.button(f"Analyze with {selected_model}", type="primary"):
    if text_input and text_input != "Enter the title followed by the full text of a news article (as the models were trained on the combined text).":
        with st.spinner(f"Analyzing text using {selected_model}..."):
            
            # Select the correct prediction function and assets
            if selected_model == 'LSTM':
                # LSTM prediction
                result, probability = predict_lstm(
                    models['LSTM'], 
                    preprocessors['lstm_tokenizer'], 
                    preprocessors['lstm_max_len'], 
                    text_input
                )
                
                # Display results
                st.success("Analysis Complete!")
                col1, col2 = st.columns(2)
                col1.metric("Prediction", result, delta="Best Performing Model")
                col2.metric("True News Probability", f"{probability*100:.2f}%")
                
            else:
                # Classical Model Prediction
                result = predict_classical(
                    models[selected_model], 
                    preprocessors['tfidf_vectorizer'], 
                    text_input
                )
                
                # Display results
                st.success("Analysis Complete!")
                st.metric("Prediction", result)

    else:
        st.warning("Please enter a news article text to begin analysis.")
        
st.markdown("---")

st.caption("Application deployed on Streamlit.io. Models trained from the provided Jupyter Notebook.")