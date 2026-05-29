import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset (Make sure CSV is in same folder)
data = pd.read_csv('mail_data.csv')

# 2. Replace null values
data = data.where((pd.notnull(data)), '')

# 3. Clean + Label Encoding (FIXED)
data['Category'] = data['Category'].str.strip().str.lower()
data['Category'] = data['Category'].map({'spam': 0, 'ham': 1})

# Remove any rows with missing labels (if mapping failed anywhere)
data = data.dropna(subset=['Category'])

# Convert to integer
data['Category'] = data['Category'].astype(int)

# 4. Separate data
X = data['Message']
Y = data['Category']

# 5. Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=3
)

# 6. Convert text to numbers
feature_extraction = TfidfVectorizer(
    min_df=1,
    stop_words='english',
    lowercase=True
)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# 7. Train Model (FIX: balanced to avoid always ham)
model = LogisticRegression(class_weight='balanced')
model.fit(X_train_features, Y_train)

# 8. Accuracy
prediction_train = model.predict(X_train_features)
print("Training Accuracy:", accuracy_score(Y_train, prediction_train))

prediction_test = model.predict(X_test_features)
print("Test Accuracy:", accuracy_score(Y_test, prediction_test))

# 9. Detailed Report
print("\nClassification Report:\n")
print(classification_report(Y_test, prediction_test))

# 10. Test with multiple inputs
test_mails = [
    "Congratulations! You won a free lottery ticket",
    "Meeting at 5 pm tomorrow",
    "Get free recharge now!!!",
    "Your order has been delivered"
]

input_data_features = feature_extraction.transform(test_mails)
predictions = model.predict(input_data_features)

print("\nPredictions:\n")
for mail, pred in zip(test_mails, predictions):
    if pred == 0:
        print(f"'{mail}' → Spam Mail ❌")
    else:
        print(f"'{mail}' → Ham Mail ✅")