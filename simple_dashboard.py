import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

# Page config
st.set_page_config(page_title="Email Classifier", page_icon="📧")

# Load models
@st.cache_resource
def load_models():
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('svm_tfidf.pkl', 'rb') as f:
            svm_model = pickle.load(f)
        return vectorizer, svm_model
    except:
        st.error("❌ Model files not found! Run 'python train_models.py' first.")
        return None, None

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return ' '.join(text.split())

# Main app
st.title("📧 Email Classifier Dashboard")

# Load models
vectorizer, svm_model = load_models()

if vectorizer and svm_model:
    # Email input
    email = st.text_area("Enter email content:", height=200)
    
    if st.button("Classify Email"):
        if email.strip():
            # Preprocess and predict
            processed = preprocess_text(email)
            X = vectorizer.transform([processed])
            prediction = svm_model.predict(X)[0]
            
            # Get confidence using decision function (distance from hyperplane)
            try:
                # Try to get probabilities if available
                confidence = max(svm_model.predict_proba(X)[0])
            except AttributeError:
                # For LinearSVC, use decision function and convert to confidence
                decision_scores = svm_model.decision_function(X)[0]
                # For multi-class, use softmax-like transformation
                exp_scores = np.exp(decision_scores)
                probabilities = exp_scores / np.sum(exp_scores)
                confidence = max(probabilities)
            
            # Display result
            st.success(f"📧 Classification: {prediction}")
            st.info(f"Confidence: {confidence:.1%}")
            
            # Color coding
            if prediction == "Spam":
                st.error("🚫 This appears to be spam!")
            elif prediction == "Important":
                st.warning("⚠️ This appears to be important!")
            else:
                st.success("📧 This appears to be a work email.")
        else:
            st.warning("Please enter some text to classify.")
    
    # Model info
    st.markdown("---")
    st.markdown("### Model Information")
    st.markdown("- **Model**: SVM with TF-IDF")
    st.markdown("- **Accuracy**: 99.86%")
    st.markdown("- **Categories**: Work, Important, Spam")
else:
    st.error("Please ensure model files are available and run 'python train_models.py' if needed.")

