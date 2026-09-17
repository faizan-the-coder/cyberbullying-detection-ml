import pandas as pd
import re
import sqlite3

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import gradio as gr

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
# 3. Train/Test Split
# =========================

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# 4. TF-IDF
# =========================

vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# =========================
# 5. Train Model
# =========================

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

# =========================
# 6. Evaluate Model
# =========================

predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy*100:.2f}%")

# =========================
# 7. SQLite Database
# =========================

conn = sqlite3.connect(
    "cyberbullying_logs.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    comment TEXT,
    prediction TEXT
)
""")

conn.commit()

# =========================
# 8. Prediction Function
# =========================

def predict_comment(comment):

    cleaned = clean_text(comment)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    if prediction == 1:
        result = "Cyberbullying Detected"
    else:
        result = "Safe Comment"

    # Save to database

    cursor.execute(
        "INSERT INTO logs(comment,prediction) VALUES (?,?)",
        (comment, result)
    )

    conn.commit()

    return result

# =========================
# 9. Gradio Interface
# =========================

app = gr.Interface(
    fn=predict_comment,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Enter a social media comment..."
    ),
    outputs="text",
    title="Cyberbullying Detection System",
    description="Detect whether a comment is cyberbullying or safe."
)

# =========================
# 10. Launch App
# =========================

app.launch()