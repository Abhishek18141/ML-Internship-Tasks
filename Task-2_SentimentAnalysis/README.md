# Sentiment Analysis using Logistic Regression

## Overview
This project performs **Sentiment Analysis** on tweets using **TF-IDF Vectorization** and **Logistic Regression**. The model predicts whether a tweet is **positive** or **negative** based on text features.

## Dataset
- **twitter_sentiment.csv** (Sentiment140 Dataset)
- Contains **text** (tweets) and **target labels**:
  - **0** → Negative sentiment
  - **1** → Positive sentiment

## Implementation Steps
1. **Load the Dataset**: Read and preprocess tweet data.
2. **Preprocess Text Data**:
   - Remove URLs, mentions, punctuation.
   - Convert to lowercase.
   - Remove stopwords.
3. **Convert Text to Features**:
   - Use **TF-IDF Vectorization** (max 5000 features).
4. **Train a Logistic Regression Model**:
   - Fit on training data.
   - Predict on test data.
5. **Evaluate Model Performance**:
   - Accuracy Score
   - Classification Report
   - Confusion Matrix
6. **Save the Model and Vectorizer**:
   - `sentiment_model.pkl` → Saved Logistic Regression model.
   - `tfidf_vectorizer.pkl` → Saved TF-IDF vectorizer.

## Files in This Repository
- `sentiment_analysis.ipynb` - Jupyter Notebook with full implementation.
- `twitter_sentiment.csv` - Dataset containing tweets and sentiment labels.
- `sentiment_model.pkl` - Saved trained Logistic Regression model.
- `tfidf_vectorizer.pkl` - Saved TF-IDF vectorizer.
- `README.md` - Documentation for the project.

## How to Run
1. **Install dependencies**:
   ```sh
   pip install pandas numpy matplotlib seaborn nltk scikit-learn joblib
   ```
2. **Run the Python script or Jupyter Notebook**:
   ```sh
   python sentiment_analysis.py
   ```
3. **Example usage**:
   ```python
   import joblib
   model = joblib.load("sentiment_model.pkl")
   vectorizer = joblib.load("tfidf_vectorizer.pkl")
   sample_text = ["I love this product! It's amazing."]
   sample_features = vectorizer.transform(sample_text)
   prediction = model.predict(sample_features)
   print("Predicted Sentiment:", "Positive" if prediction[0] == 1 else "Negative")
   ```

## Results
- Achieved **high accuracy** on test data.
- Model correctly classifies positive and negative tweets.

## Future Enhancements
- Implement **Deep Learning** for better text classification.
- Add **Neutral Sentiment** category.
- Deploy as a **Flask/Streamlit web app**.

## Author
**Abhishek Pandey**  
📅 Date: 23-02-2025

