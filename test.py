import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# =========================
# 1. Load Dataset
# =========================

df = pd.read_csv("cyberbullying.csv")

# =========================
# 2. Text Cleaning
# =========================

def clean_text(text):
    text = str(text).lower()

    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text.strip()

df["text"] = df["text"].apply(clean_text)

# =========================
# 3. Prepare Data
# =========================

X = df["text"]
y = df["label"]

# =========================
# 4. Convert Text to Numbers
# =========================

vectorizer = TfidfVectorizer()

X_tfidf = vectorizer.fit_transform(X)

# =========================
# 5. Train Model
# =========================

model = LogisticRegression(max_iter=1000)

model.fit(X_tfidf, y)

print("Model trained successfully!")

# =========================
# 6. Prediction Loop
# =========================

while True:

    comment = input("\nEnter Comment (or type exit): ")

    if comment.lower() == "exit":
        break

    cleaned_comment = clean_text(comment)

    vector = vectorizer.transform([cleaned_comment])

    prediction = model.predict(vector)[0]

    if prediction == 1:
        print("Cyberbullying Detected")
    else:
        print("Safe Comment")