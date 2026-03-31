# ==========================================
# CUSTOMER SUPPORT TICKET CLASSIFICATION
# ==========================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# ------------------------------------------
# 2. Load Dataset
# ------------------------------------------

df = pd.read_csv(r"C:\Users\pavit\OneDrive\Documents\Future Intern\task_2\customer_support_tickets.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ------------------------------------------
# 3. Combine Subject + Description
# ------------------------------------------

df['text'] = df['Ticket Subject'] + " " + df['Ticket Description']

# ------------------------------------------
# 4. Text Cleaning Function
# ------------------------------------------

stop_words = set(stopwords.words('english'))

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    tokens = word_tokenize(text)

    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

print("\nCleaned Text Example:")
print(df['clean_text'].head())

# ------------------------------------------
# 5. Convert Text to Numerical Features
# ------------------------------------------

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df['clean_text'])

# ------------------------------------------
# 6. CATEGORY CLASSIFICATION
# ------------------------------------------

y_category = df['Ticket Type']

X_train, X_test, y_train, y_test = train_test_split(
    X, y_category, test_size=0.2, random_state=42
)

category_model = LogisticRegression(max_iter=1000)

category_model.fit(X_train, y_train)

category_predictions = category_model.predict(X_test)

print("\n===== Ticket Category Classification =====")

print("Accuracy:", accuracy_score(y_test, category_predictions))

print(classification_report(y_test, category_predictions))

# ------------------------------------------
# 7. PRIORITY PREDICTION
# ------------------------------------------

y_priority = df['Ticket Priority']

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y_priority, test_size=0.2, random_state=42
)

priority_model = LogisticRegression(max_iter=1000)

priority_model.fit(X_train2, y_train2)

priority_predictions = priority_model.predict(X_test2)

print("\n===== Ticket Priority Classification =====")

print("Accuracy:", accuracy_score(y_test2, priority_predictions))

print(classification_report(y_test2, priority_predictions))

# ------------------------------------------
# 8. Test with a New Ticket
# ------------------------------------------

def predict_ticket(ticket_text):

    cleaned = clean_text(ticket_text)

    vector = vectorizer.transform([cleaned])

    category = category_model.predict(vector)[0]

    priority = priority_model.predict(vector)[0]

    print("\nPredicted Category:", category)
    print("Predicted Priority:", priority)


# Example ticket
test_ticket = "I was charged twice for my subscription payment"

predict_ticket(test_ticket)

# ------------------------------------------
# 9. Visualization of Ticket Categories
# ------------------------------------------

plt.figure(figsize=(6, 4))  # Smaller figure size

df['Ticket Type'].value_counts().plot(kind='bar')

plt.title("Distribution of Ticket Categories")

plt.xlabel("Ticket Type")

plt.ylabel("Count")

plt.show()

# ------------------------------------------
# 10. Visualization of Ticket Priority
# ------------------------------------------

plt.figure(figsize=(6, 4))  # Smaller figure size

df['Ticket Priority'].value_counts().plot(kind='bar')

plt.title("Distribution of Ticket Priority")

plt.xlabel("Priority Level")

plt.ylabel("Count")

plt.show()