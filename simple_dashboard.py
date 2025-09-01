import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

import torch
from transformers import BertTokenizer, BertModel

# ===========================
# Sidebar - File Upload
# ===========================
st.sidebar.title("📩 Email Spam Classifier")
uploaded_file = st.sidebar.file_uploader("Upload cleaned email dataset (CSV)", type=["csv"])

if uploaded_file:
    # ===========================
    # Load and Preview Data
    # ===========================
    df = pd.read_csv(uploaded_file)
    st.title("📊 Email Dataset Overview")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

    X = df["text"].astype(str)
    y = df["label"]

    # ===========================
    # Feature Extraction Choice
    # ===========================
    feature_choice = st.sidebar.radio("Choose Feature Representation:", ["TF-IDF", "BERT"])

    if feature_choice == "TF-IDF":
        tfidf = TfidfVectorizer(max_features=5000)
        X_features = tfidf.fit_transform(X)
    else:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased")

        @st.cache_resource
        def get_bert_embeddings(texts):
            embeddings = []
            for t in texts:
                inputs = tokenizer(t, return_tensors="pt", truncation=True, padding=True, max_length=128)
                with torch.no_grad():
                    outputs = bert_model(**inputs)
                vec = outputs.last_hidden_state[:,0,:].numpy()[0]
                embeddings.append(vec)
            return np.array(embeddings)

        sample_size = min(2000, len(df))
        X_features = get_bert_embeddings(X.head(sample_size))
        y = y.head(sample_size)

    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

    # ===========================
    # Model Selection
    # ===========================
    model_choice = st.sidebar.radio("Choose a Model:", ["Naive Bayes", "Linear SVC"])

    if model_choice == "Naive Bayes":
        model = MultinomialNB()
    else:
        model = LinearSVC()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ===========================
    # Results
    # ===========================
    st.header("🔍 Model Evaluation")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    x=["Ham", "Spam"], y=["Ham", "Spam"], title="Confusion Matrix")
    st.plotly_chart(fig)

    # ===========================
    # Word Cloud Visualization
    # ===========================
    if feature_choice == "TF-IDF":
        st.header("☁ Word Cloud of Emails")
        from wordcloud import WordCloud
        wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(X))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

    # ===========================
    # Try it out
    # ===========================
    st.header("✉ Test with Your Own Email")
    user_input = st.text_area("Paste your email text here:")
    if st.button("Classify"):
        if feature_choice == "TF-IDF":
            vec = tfidf.transform([user_input])
        else:
            vec = get_bert_embeddings([user_input])
        pred = model.predict(vec)[0]
        st.success(f"Prediction: **{pred}**")
else:
    st.info("👆 Please upload your CSV file with 'text' and 'label' columns to begin.")
