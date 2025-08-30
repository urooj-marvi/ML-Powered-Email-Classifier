"""
Script to train and save ML models for the Email Classifier Dashboard
"""

import pandas as pd
import numpy as np
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def preprocess_text(text):
    """Preprocess text for classification"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def train_and_save_models():
    """Train models and save them to disk"""
    print("Loading dataset...")
    
    try:
        # Load the dataset
        df = pd.read_csv('emails_cleaned.csv')
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        
        # Preprocess the data
        print("Preprocessing text data...")
        df['processed_body'] = df['body'].apply(preprocess_text)
        
        # Remove rows with empty processed text
        df = df[df['processed_body'].str.len() > 0]
        print(f"After preprocessing, dataset shape: {df.shape}")
        
        # Create TF-IDF vectorizer
        print("Creating TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(
            max_features=5000, 
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Fit and transform the data
        X = vectorizer.fit_transform(df['processed_body'])
        y = df['label']
        
        print(f"TF-IDF features shape: {X.shape}")
        print(f"Number of classes: {len(y.unique())}")
        print(f"Class distribution:\n{y.value_counts()}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Train Naive Bayes
        print("\nTraining Naive Bayes model...")
        nb_model = MultinomialNB()
        nb_model.fit(X_train, y_train)
        
        # Evaluate Naive Bayes
        y_pred_nb = nb_model.predict(X_test)
        nb_accuracy = accuracy_score(y_test, y_pred_nb)
        print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")
        print("Naive Bayes Classification Report:")
        print(classification_report(y_test, y_pred_nb))
        
        # Train SVM
        print("\nTraining SVM model...")
        svm_model = LinearSVC(random_state=42, max_iter=1000)
        svm_model.fit(X_train, y_train)
        
        # Evaluate SVM
        y_pred_svm = svm_model.predict(X_test)
        svm_accuracy = accuracy_score(y_test, y_pred_svm)
        print(f"SVM Accuracy: {svm_accuracy:.4f}")
        print("SVM Classification Report:")
        print(classification_report(y_test, y_pred_svm))
        
        # Save models
        print("\nSaving models...")
        
        # Save TF-IDF vectorizer
        with open('tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        print("✓ TF-IDF vectorizer saved")
        
        # Save Naive Bayes model
        with open('naive_bayes_tfidf.pkl', 'wb') as f:
            pickle.dump(nb_model, f)
        print("✓ Naive Bayes model saved")
        
        # Save SVM model
        with open('svm_tfidf.pkl', 'wb') as f:
            pickle.dump(svm_model, f)
        print("✓ SVM model saved")
        
        print("\n🎉 All models trained and saved successfully!")
        print(f"Best performing model: SVM with {svm_accuracy:.4f} accuracy")
        
        return True
        
    except FileNotFoundError:
        print("❌ Error: emails_cleaned.csv not found!")
        print("Please ensure the dataset file is in the same directory.")
        return False
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Email Classifier Model Training")
    print("=" * 40)
    
    success = train_and_save_models()
    
    if success:
        print("\n✅ You can now run the dashboard with:")
        print("   streamlit run email_classifier_dashboard.py")
    else:
        print("\n❌ Model training failed. Please check the error messages above.")
