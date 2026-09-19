# 🚨 Comment Toxicity Detection Using Deep Learning

## 📌 Project Overview

**Comment Toxicity Detection** is an NLP-based deep learning project designed to identify different types of toxic behavior in online comments.

The project treats toxicity detection as a **multi-label text classification problem**, where a single comment can belong to one or more toxicity categories.

The implemented system compares two deep learning models:

* **Convolutional Neural Network (CNN)**
* **Bidirectional Long Short-Term Memory (BiLSTM)**

The model predicts the following six categories:

* `toxic`
* `severe_toxic`
* `obscene`
* `threat`
* `insult`
* `identity_hate`

The supplied validation run reports a mean ROC-AUC of **0.9735 for CNN** and **0.9720 for LSTM**. The code selects CNN based on the higher mean ROC-AUC in this validation run.

---

## 🎯 Objectives

The main objectives of this project are:

1. Clean and preprocess raw comment text.
2. Convert text into numerical sequences using tokenization.
3. Train deep learning models for toxicity classification.
4. Compare CNN and Bidirectional LSTM architectures.
5. Evaluate model performance using validation metrics and ROC-AUC.
6. Save the selected model and tokenizer for future prediction.

---


# 🔮 Future Improvements

Possible extensions include:

* Use transformer-based models such as BERT.
* Handle class imbalance explicitly.
* Add precision, recall and F1-score.
* Tune classification thresholds separately for each label.
* Deploy the trained model through Streamlit.
* Use the modern Keras `.keras` model format instead of legacy HDF5.
* Add monitoring and logging for deployed predictions.

---

# 👩‍💻 Project Information

**Project:** Comment Toxicity Detection
**Domain:** Natural Language Processing / Deep Learning
**Task:** Multi-Label Text Classification
**Models:** CNN and Bidirectional LSTM
**Framework:** TensorFlow / Keras
**Language:** Python

---

## ⭐ Acknowledgement

This README is based on the supplied Comment Toxicity project notebook and its recorded outputs.
