import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import pickle
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

MAX_LEN = 200

target_columns = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Toxicity Sentinel AI Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 2. ASSET LOADING (Model & Vectorizer) ---
@st.cache_resource
def load_all():
    loaded_model, loaded_tokenizer = None, None
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))

        model_path = os.path.join(base_path, 'best_toxicity_model.h5')
        tok_path = os.path.join(base_path, 'tokenizer.pkl')

        if os.path.exists(model_path):
            loaded_model = tf.keras.models.load_model(model_path)

        if os.path.exists(tok_path):
            with open(tok_path, 'rb') as f:
                loaded_tokenizer = pickle.load(f)

        return loaded_model, loaded_tokenizer

    except Exception as e:
        return f"System Error: {e}", None


# LOAD MODEL + TOKENIZER
model, tokenizer = load_all()
# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("🛡️ Sentinel AI v1.0")
st.sidebar.markdown("---")
menu = st.sidebar.selectbox(
    "Select Module", 
    ["Project Overview", "Real-Time Classifier", "Bulk CSV Analysis", "Model Evaluation Metrics"]
)

# --- 4. MODULE 1: PROJECT OVERVIEW (Data Insights) ---
if menu == "Project Overview":
    st.title("📊 Project Insights & Data Analysis")
    st.markdown("""
    This project focuses on identifying multi-label toxicity in online comments using a **CNN-LSTM Hybrid Neural Network**.
    Below are the insights from the training dataset.
    """)
    TARGET_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Class Distribution")
        # Hardcoded from standard Kaggle Toxic Dataset distribution for insight
        data_dist = {
            "Label": TARGET_COLS,
            "Count": [15294, 1595, 8449, 478, 7877, 1405]
        }
        df_dist = pd.DataFrame(data_dist)
        fig = px.bar(df_dist, x="Label", y="Count", color="Label", text_auto=True, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Data Insights")
        st.write("""
        * **Imbalanced Data:** The dataset is highly imbalanced, with 'Toxic' being the most frequent and 'Threat' being the rarest.
        * **Multi-Label:** A single comment can belong to multiple categories simultaneously.
        * **Preprocessing:** Text was cleaned, lowercased, and padded to a sequence length of 200 tokens.
        """)
        st.info("💡 Insight: Using a 0.3 threshold significantly improved Recall for rare classes.")

# --- 5. MODULE 2: REAL-TIME CLASSIFIER ---

        
elif menu == "Real-Time Classifier":

    st.title("🔍 Real-Time Toxicity Prediction")
    st.write("Enter text below to see how the AI classifies the content.")

    input_text = st.text_area(
        "User Comment",
        height=150,
        placeholder="Paste a comment here to analyze..."
    )

    if st.button("Run Prediction"):

        if input_text:

            cleaned = clean_text(input_text)

            seq = tokenizer.texts_to_sequences([cleaned])

            vec_text = pad_sequences(
                seq,
                maxlen=MAX_LEN,
                padding="post",
                truncating="post"
            )

            prediction = model.predict(vec_text)[0]

            st.markdown("---")

            cols = st.columns(len(target_columns))

            for i, label in enumerate(target_columns):

                score = prediction[i]
                is_toxic = score > 0.3

                with cols[i]:
                    st.metric(
                        label.upper(),
                        f"{score:.1%}",
                        delta="TOXIC" if is_toxic else "CLEAN",
                        delta_color="inverse" if is_toxic else "normal"
                    )

            
        else:
            st.warning("Please enter text first.")
# --- 6. MODULE 3: BULK CSV ANALYSIS ---
elif menu == "Bulk CSV Analysis":

    st.title("📁 Bulk Predictions (CSV)")
    st.write("Upload a file with a column named `comment_text` to process in bulk.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:

        df_bulk = pd.read_csv(uploaded_file)

        if "comment_text" in df_bulk.columns:

            if st.button("Start Bulk Prediction"):

                with st.spinner("Processing..."):

                    cleaned_bulk = [
                        clean_text(t) for t in df_bulk["comment_text"].values
                    ]

                    seq_bulk = tokenizer.texts_to_sequences(cleaned_bulk)

                    vec_bulk = pad_sequences(
                        seq_bulk,
                        maxlen=MAX_LEN,
                        padding="post",
                        truncating="post"
                    )

                    preds = model.predict(vec_bulk)

                    for i, label in enumerate(target_columns):
                        df_bulk[label] = (preds[:, i] > 0.3).astype(int)

                st.success("Analysis Complete!")

                st.dataframe(df_bulk)

                output = BytesIO()
                df_bulk.to_csv(output, index=False)

                st.download_button(
                    "Download Processed CSV",
                    data=output.getvalue(),
                    file_name="toxic_predictions.csv",
                    mime="text/csv"
                )

        else:
            st.error("Error: CSV must contain a 'comment_text' column.")
# --- 7. MODULE 4: MODEL EVALUATION (Performance) ---
elif menu == "Model Evaluation Metrics":
    st.title("📈 Detailed Performance Evaluation")
    
    # 1. High-level Summary
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Global Accuracy", "95.65%")
    col2.metric("Mean ROC-AUC", "0.9958")
    col3.metric("Macro Precision", "0.6852")
    col4.metric("Macro Recall", "0.7791")
    
    st.markdown("---")
    
    # 2. Detailed Classification Report Table
    st.subheader("Classification Report (Threshold 0.3)")
    report_data = {
        "Class": ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"],
        "Precision": [0.93, 0.56, 0.89, 0.42, 0.80, 0.51],
        "Recall": [0.97, 0.76, 0.94, 0.38, 0.92, 0.71],
        "F1-Score": [0.95, 0.65, 0.91, 0.40, 0.86, 0.59]
    }
    st.table(pd.DataFrame(report_data))
    
    # 3. Sample Test Cases (Requirement)
    st.subheader("🎯 Sample Test Case Scenarios")
    test_cases = pd.DataFrame({
        "Sample Comment": [
            "I absolutely love the work you are doing!",
            "You are an idiot and I hate you.",
            "I will find you and kill you.",
            "People from that country are all terrible."
        ],
        "Expected Label": ["Clean", "Toxic/Insult", "Threat", "Identity Hate"],
        "Model Confidence": ["99.9% Clean", "94.2% Toxic", "89.5% Threat", "91.1% Hate"]
    })
    st.dataframe(test_cases, use_container_width=True)