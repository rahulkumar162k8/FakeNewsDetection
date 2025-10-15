<<<<<<< HEAD
# AIML Fake News Detection Streamlit App

This repository contains the necessary files to deploy a multi-model Fake News Detection application on Streamlit.io.

## Models Included

The application uses five models trained from the initial Jupyter Notebook:
1.  **LSTM** (Long Short-Term Memory Network) - Identified as the best-performing model.
2.  **Logistic Regression**
3.  **Support Vector Machine (SVM)**
4.  **Multinomial Naive Bayes**
5.  **Random Forest**

## 🚀 Pre-Deployment Steps (Run Locally)

**IMPORTANT:** You must run the `model_trainer.py` script locally *before* deploying to Streamlit. This trains the models and saves the necessary assets into the `models/` directory.

1.  **Place Data:** Ensure your `Fake.csv` and `True.csv` files are in the same directory as these script files.
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run Trainer:**
    ```bash
    python model_trainer.py
    ```
    This script will:
    * Download necessary NLTK data (stopwords, wordnet).
    * Load and preprocess the data.
    * Train the TF-IDF Vectorizer and all four classical models (`.pkl` files).
    * Train the Keras Tokenizer (`.pkl` file) and the LSTM model (`.h5` file).
    * Save all these assets into a newly created **`models/`** directory.

## ☁️ Deployment on Streamlit.io

1.  **Create a GitHub Repository:** Commit all the generated files (`app.py`, `utils.py`, `requirements.txt`, `README.md`) and the newly created **`models/`** directory (containing the `.pkl` and `.h5` files) to a new GitHub repository.
2.  **Deploy via Streamlit:**
    * Go to the [Streamlit Deployment Page](https://share.streamlit.io/).
    * Click "New App".
    * Select your GitHub repository and the branch where you committed the files.
    * Set the **Main file path** to `app.py`.
=======
# AIML Fake News Detection Streamlit App

This repository contains the necessary files to deploy a multi-model Fake News Detection application on Streamlit.io.

## Models Included

The application uses five models trained from the initial Jupyter Notebook:
1.  **LSTM** (Long Short-Term Memory Network) - Identified as the best-performing model.
2.  **Logistic Regression**
3.  **Support Vector Machine (SVM)**
4.  **Multinomial Naive Bayes**
5.  **Random Forest**

## 🚀 Pre-Deployment Steps (Run Locally)

**IMPORTANT:** You must run the `model_trainer.py` script locally *before* deploying to Streamlit. This trains the models and saves the necessary assets into the `models/` directory.

1.  **Place Data:** Ensure your `Fake.csv` and `True.csv` files are in the same directory as these script files.
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run Trainer:**
    ```bash
    python model_trainer.py
    ```
    This script will:
    * Download necessary NLTK data (stopwords, wordnet).
    * Load and preprocess the data.
    * Train the TF-IDF Vectorizer and all four classical models (`.pkl` files).
    * Train the Keras Tokenizer (`.pkl` file) and the LSTM model (`.h5` file).
    * Save all these assets into a newly created **`models/`** directory.

## ☁️ Deployment on Streamlit.io

1.  **Create a GitHub Repository:** Commit all the generated files (`app.py`, `utils.py`, `requirements.txt`, `README.md`) and the newly created **`models/`** directory (containing the `.pkl` and `.h5` files) to a new GitHub repository.
2.  **Deploy via Streamlit:**
    * Go to the [Streamlit Deployment Page](https://share.streamlit.io/).
    * Click "New App".
    * Select your GitHub repository and the branch where you committed the files.
    * Set the **Main file path** to `app.py`.
>>>>>>> 7d4d2e3e66fa11c460251441040bdb763edfdc55
    * Click "Deploy!"