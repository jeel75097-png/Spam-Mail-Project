# 📧 Spam Mail Detection System using Machine Learning

A Machine Learning project that classifies emails and messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and Logistic Regression.

---

## 📌 Project Overview

Spam messages are unwanted emails or text messages that often contain advertisements, scams, phishing attempts, or malicious links.

This project uses Machine Learning and NLP techniques to automatically identify whether a message is spam or legitimate.

---

## 🎯 Objectives

- Detect spam emails automatically
- Convert text data into machine-readable format
- Apply Natural Language Processing (NLP)
- Train a classification model
- Evaluate model performance
- Predict spam messages in real time

---

## 📂 Dataset

Dataset: **mail_data.csv**

### Features

| Column | Description |
|----------|------------|
| Category | Spam or Ham |
| Message | Email/SMS Content |

### Target Variable

```text
Category
```

- Spam → 0
- Ham → 1

---

## 🛠 Technologies Used

- Python
- NumPy
- Pandas
- Scikit-Learn
- NLP (TF-IDF Vectorization)

---

## 🤖 Machine Learning Algorithm

### Logistic Regression

The project uses Logistic Regression for binary classification.

Advantages:

- Fast training
- High accuracy for text classification
- Efficient for large datasets
- Easy to interpret

---

## 🚀 Project Workflow

### 1️⃣ Load Dataset

Read dataset using Pandas.

```python
pd.read_csv("mail_data.csv")
```

---

### 2️⃣ Data Cleaning

- Replace null values
- Remove invalid records
- Normalize labels

Convert:

```text
spam → 0
ham → 1
```

---

### 3️⃣ Feature Selection

Separate:

```text
Message → Features (X)
Category → Target (Y)
```

---

### 4️⃣ Train-Test Split

Split dataset into:

- 80% Training Data
- 20% Testing Data

---

### 5️⃣ Text Vectorization

Convert text into numerical vectors using:

```python
TfidfVectorizer()
```

Benefits:

- Removes common words
- Highlights important keywords
- Improves model performance

---

### 6️⃣ Model Training

Train a Logistic Regression model:

```python
LogisticRegression(
    class_weight='balanced'
)
```

Using balanced weights helps improve spam detection.

---

### 7️⃣ Model Evaluation

Evaluate using:

- Training Accuracy
- Testing Accuracy
- Classification Report

---

### 8️⃣ Real-Time Predictions

The model predicts whether custom messages are:

```text
Spam Mail ❌
```

or

```text
Ham Mail ✅
```

---

## 📊 NLP Concepts Used

### TF-IDF Vectorization

TF-IDF converts text into numerical vectors.

Benefits:

- Identifies important words
- Reduces impact of common words
- Improves classification accuracy

---

## 📈 Evaluation Metrics

### Accuracy Score

Measures overall prediction performance.

### Classification Report

Provides:

- Precision
- Recall
- F1 Score
- Support

---

## 📁 Project Structure

```text
Spam-Mail-Detection/
│
├── mail_data.csv
├── spam.py
├── README.md
├── requirements.txt
│
└── screenshots/
```

---

## 📦 Installation

Install required libraries:

```bash
pip install numpy pandas scikit-learn
```

---

## ▶️ How to Run

Clone the repository:

```bash
git clone https://github.com/yourusername/spam-mail-detection.git
```

Move into the project directory:

```bash
cd spam-mail-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python spam.py
```

---

## 💻 Example Predictions

### Input Messages

```text
Congratulations! You won a free lottery ticket
Meeting at 5 pm tomorrow
Get free recharge now!!!
Your order has been delivered
```

### Output

```text
'Congratulations! You won a free lottery ticket'
→ Spam Mail ❌

'Meeting at 5 pm tomorrow'
→ Ham Mail ✅

'Get free recharge now!!!'
→ Spam Mail ❌

'Your order has been delivered'
→ Ham Mail ✅
```

---

## 🔮 Future Improvements

- Streamlit Web Application
- Email Integration
- SMS Spam Detection
- Deep Learning Models (LSTM/BERT)
- Real-Time Gmail Filtering
- API Deployment

---

## 📈 Applications

- Email Spam Filters
- SMS Filtering Systems
- Cybersecurity Solutions
- Fraud Detection Systems
- Enterprise Email Security

---

## 👨‍💻 Author

Developed as a Machine Learning project for spam message classification and NLP-based text analysis.

---

## ⭐ Support

If you found this project useful, please give it a ⭐ on GitHub.
