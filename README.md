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

## 🗂️ Dataset

The project uses `train.csv` and `test.csv`.

### Training Dataset

Before preprocessing:

* Rows: **159,571**
* Columns: **8**

Columns:

```text
id
comment_text
toxic
severe_toxic
obscene
threat
insult
identity_hate
```

### Test Dataset

Before preprocessing:

* Rows: **153,164**
* Columns: **2**

Columns:

```text
id
comment_text
```

After duplicate removal:

```text
Training data: 158,220 comments
Test data:     150,802 comments
```

---

## 🧠 Problem Type

This project uses **multi-label classification**.

For example, one comment may simultaneously be:

```text
toxic = 1
insult = 1
obscene = 0
threat = 0
```

Therefore, six independent outputs are generated for every comment.

---

## 🛠️ Technologies Used

| Technology          | Purpose                       |
| ------------------- | ----------------------------- |
| Python              | Programming language          |
| Pandas              | Data manipulation             |
| NumPy               | Numerical processing          |
| Matplotlib          | Visualization                 |
| Seaborn             | Data visualization            |
| Regular Expressions | Text cleaning                 |
| NLTK                | Natural language processing   |
| TensorFlow          | Deep learning                 |
| Keras               | Neural network implementation |
| Scikit-learn        | Data splitting and evaluation |

---

## 🔄 Project Workflow

```text
Raw Comments
     ↓
Load Dataset
     ↓
Data Inspection
     ↓
Remove ID Column
     ↓
Text Cleaning
     ↓
Remove Duplicates
     ↓
Tokenization
     ↓
Sequence Padding
     ↓
Train / Validation Split
     ↓
 ┌───────────────┐
 │               │
CNN             BiLSTM
 │               │
 └───────┬───────┘
         ↓
Six Toxicity Scores
         ↓
Model Evaluation
         ↓
Model Comparison
         ↓
Save Selected Model
```

---

# 🧹 Data Preprocessing

## 1. Remove ID

The `id` column is not required for text classification.

```python
train_df = train_df.drop(columns=['id'], errors='ignore')
test_df = test_df.drop(columns=['id'], errors='ignore')
```

## 2. Text Cleaning

The project uses the following cleaning function:

```python
def clean(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

### Cleaning operations

* Convert text to lowercase
* Remove URLs
* Remove non-alphabetic characters
* Remove unnecessary spaces

---

## 3. Duplicate Removal

Duplicate comments are removed:

```python
train_df.drop_duplicates(
    subset=['comment_text'],
    inplace=True
)

test_df.drop_duplicates(
    subset=['comment_text'],
    inplace=True
)
```

The source code reports:

```text
Duplicates removed from training: 1,351
Final training shape: 158,220 × 7

Duplicates removed from test: 2,362
Final test shape: 150,802 × 1
```

---

# 🔤 Tokenization

The project converts text into numerical sequences using Keras `Tokenizer`.

```python
MAX_WORDS = 20000
MAX_LEN = 150

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(
    train_df['comment_text']
)
```

### Parameters

| Parameter               |   Value |
| ----------------------- | ------: |
| Maximum vocabulary      |  20,000 |
| Maximum sequence length |     150 |
| OOV token               | `<OOV>` |

The sequences are padded:

```python
train_sequences = tokenizer.texts_to_sequences(
    train_df['comment_text']
)

X_train = pad_sequences(
    train_sequences,
    maxlen=MAX_LEN,
    padding='post'
)
```

Final shapes reported by the code:

```text
X_train = (158220, 150)
X_test  = (150802, 150)
```

---

# 🎯 Target Variables

The model predicts six output labels:

```python
target_cols = [
    'toxic',
    'severe_toxic',
    'obscene',
    'threat',
    'insult',
    'identity_hate'
]
```

Target matrix:

```python
y_train = train_df[target_cols].values
```

Shape:

```text
(158220, 6)
```

---

# ✂️ Train / Validation Split

The training data is divided into training and validation sets:

```python
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.1,
    random_state=42
)
```

Result:

```text
Training samples:   142,398
Validation samples: 15,822
```

---

# 🧠 CNN Model

The CNN architecture used in the project is:

```text
Input
  ↓
Embedding
  ↓
SpatialDropout1D
  ↓
Conv1D
  ↓
GlobalMaxPooling1D
  ↓
Dense
  ↓
Dropout
  ↓
Dense(6, Sigmoid)
```

Implementation:

```python
cnn_model = Sequential([
    Input(shape=(MAX_LEN,)),
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=128
    ),
    SpatialDropout1D(0.3),
    Conv1D(
        128,
        5,
        activation="relu"
    ),
    GlobalMaxPooling1D(),
    Dense(
        128,
        activation="relu"
    ),
    Dropout(0.3),
    Dense(
        6,
        activation="sigmoid"
    )
])
```

### Why CNN?

CNN can learn useful local patterns in text sequences through convolution filters.

---

# 🔁 Bidirectional LSTM Model

The LSTM architecture is:

```text
Input
  ↓
Embedding
  ↓
SpatialDropout1D
  ↓
Bidirectional LSTM
  ↓
GlobalMaxPooling1D
  ↓
Dense
  ↓
Dropout
  ↓
Dense(6, Sigmoid)
```

Implementation:

```python
lstm_model = Sequential([
    Input(shape=(MAX_LEN,)),
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=128
    ),
    SpatialDropout1D(0.3),
    Bidirectional(
        LSTM(
            64,
            return_sequences=True
        )
    ),
    GlobalMaxPooling1D(),
    Dense(
        128,
        activation="relu"
    ),
    Dropout(0.3),
    Dense(
        6,
        activation="sigmoid"
    )
])
```

### Why LSTM?

Bidirectional LSTM processes a sequence in both directions and is designed to capture contextual relationships between words.

---

# ⚙️ Model Training

Both models use:

```text
Optimizer : Adam
Loss      : Binary Cross-Entropy
Output    : Sigmoid
```

Example:

```python
cnn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['Accuracy']
)
```

Because this is a multi-label classification problem, six independent sigmoid outputs are used.

---

# 📊 CNN Results

The supplied code reports:

```text
Validation Loss     : 0.0499
Validation Accuracy : 0.9941
Mean ROC-AUC        : 0.9735
```

### CNN ROC-AUC by category

| Category      | ROC-AUC |
| ------------- | ------: |
| Toxic         |  0.9744 |
| Severe Toxic  |  0.9890 |
| Obscene       |  0.9877 |
| Threat        |  0.9485 |
| Insult        |  0.9819 |
| Identity Hate |  0.9597 |

---

# 📈 LSTM Results

The supplied code reports:

```text
Validation Loss     : 0.0505
Validation Accuracy : 0.9940
Mean ROC-AUC        : 0.9720
```

### LSTM ROC-AUC by category

| Category      | ROC-AUC |
| ------------- | ------: |
| Toxic         |  0.9762 |
| Severe Toxic  |  0.9882 |
| Obscene       |  0.9859 |
| Threat        |  0.9388 |
| Insult        |  0.9809 |
| Identity Hate |  0.9622 |

---

# 🔍 Model Comparison

| Metric              |    CNN | BiLSTM |
| ------------------- | -----: | -----: |
| Validation Loss     | 0.0499 | 0.0505 |
| Validation Accuracy | 0.9941 | 0.9940 |
| Mean ROC-AUC        | 0.9735 | 0.9720 |

The supplied code selects the model with the higher mean ROC-AUC for saving.

```python
best_model = (
    lstm_model
    if lstm_auc > cnn_auc
    else cnn_model
)
```

For the supplied validation run, this results in the CNN model being saved.

---

# 🔮 Prediction

A new comment can be processed using:

```python
def compare_models(text):

    cleaned = clean(text)

    seq = tokenizer.texts_to_sequences(
        [cleaned]
    )

    pad = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    lstm_probs = lstm_model.predict(pad)[0]
    cnn_probs = cnn_model.predict(pad)[0]

    comparison_df = pd.DataFrame({
        'Category': target_cols,
        'LSTM Score': [
            round(float(p), 4)
            for p in lstm_probs
        ],
        'CNN Score': [
            round(float(p), 4)
            for p in cnn_probs
        ]
    })

    return comparison_df
```

Example input:

```text
You are an absolute idiot and I hate you.
```

The model returns a score for each of the six categories.

---

# 💾 Model Saving

The project creates a folder:

```text
toxicity_model/
```

The selected model is saved as:

```python
best_model.save(
    "toxicity_model/best_toxicity_model.h5"
)
```

The tokenizer is saved using:

```python
pickle.dump(
    tokenizer,
    open(
        "toxicity_model/tokenizer.pkl",
        "wb"
    )
)
```

### Saved Files

```text
toxicity_model/
├── best_toxicity_model.h5
└── tokenizer.pkl
```

---

# 📁 Project Structure

```text
Comment-Toxicity-Detection/
│
├── train.csv
├── test.csv
├── Comment Toxicity code.ipynb
│
├── toxicity_model/
│   ├── best_toxicity_model.h5
│   └── tokenizer.pkl
│
└── README.md
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/comment-toxicity-detection.git
```

Move into the project directory:

```bash
cd comment-toxicity-detection
```

Install required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras nltk
```

---

# ▶️ Running the Project

1. Place `train.csv` and `test.csv` in the project directory.
2. Open the notebook:

```text
Comment Toxicity code.ipynb
```

3. Run the cells sequentially.
4. The notebook performs:

   * Data loading
   * Data analysis
   * Text preprocessing
   * Tokenization
   * Model training
   * Model evaluation
   * Model comparison
   * Prediction
   * Model saving

---

# 📌 Key Concepts

### Multi-Label Classification

A comment can belong to multiple categories simultaneously.

### Embedding

Converts token IDs into dense numerical vectors that can be learned by the neural network.

### CNN

Learns local patterns from sequences.

### Bidirectional LSTM

Processes sequence information from both directions.

### Sigmoid

Produces an independent output score for each toxicity category.

### Binary Cross-Entropy

Used as the loss function for the six independent binary targets.

### ROC-AUC

Used in the supplied code to measure how well each class separates positive and negative examples across thresholds.

---

# 🎓 Viva Questions

### 1. Why is this a multi-label classification problem?

Because one comment can contain more than one type of toxicity.

### 2. Why is sigmoid used?

Each of the six categories is predicted independently, so each output requires its own score.

### 3. Why is padding required?

Neural networks process batches of uniform-sized sequences, so comments are padded to a fixed length of 150.

### 4. Why remove duplicate comments?

To reduce repeated training examples.

### 5. What is the role of Embedding?

It converts integer token IDs into learnable dense vectors.

### 6. What is the difference between CNN and LSTM?

CNN focuses on local patterns, while LSTM is designed to capture sequential relationships.

### 7. Why is binary cross-entropy used?

Each toxicity category is treated as a separate binary classification target.

### 8. What evaluation metric is calculated in the model-comparison code?

The code uses `roc_auc_score()` and averages the six class-level ROC-AUC values.

---

# ⚠️ Note About the Source Code

The notebook prints some ROC-AUC results using the label **"Accuracy"**, but the calculation is performed using:

```python
roc_auc_score()
```

Therefore, these particular values are more accurately described as **ROC-AUC**, not conventional classification accuracy.

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
