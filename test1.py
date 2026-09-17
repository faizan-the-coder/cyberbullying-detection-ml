import pandas as pd
import re
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

import gradio as gr


df = pd.read_csv("cyberbullying.csv")


def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'http\S+', '', text)

    text = re.sub(r'www\.\S+', '', text)

    text = re.sub(r'@\w+', '', text)

    text = re.sub(r'\d+', '', text)

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = " ".join(text.split()) #you are stupid

    return text


df["text"] = df["text"].apply(clean_text)

X = df["text"]
vectorizer = TfidfVectorizer()

X_tfidf = vectorizer.fit_transform(X)

X=X_tfidf
y= df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, #20% testing
    random_state=42
)


model = LogisticRegression()  #machine learning model

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)
print(f"Model Accuracy: {accuracy*100:.2f}%")

#sqlite3 database
conn=sqlite3.connect('cyberbullying_db',check_same_thread=False)
cursor=conn.cursor()

cursor.execute('CREATE TABLE IF NOT EXISTS comments (id INTEGER PRIMARY KEY AUTOINCREMENT, comment TEXT, prediction TEXT)')


# while True:
def predict_comment(comment):

    # comment=input("Enter a comment ('bye' to exit): ") #you are stupid!!!
    # if comment.lower()=='bye':
    #     break

    cleaned_text=clean_text(comment)

    comment_tfidf=vectorizer.transform([cleaned_text])

    prediction=model.predict(comment_tfidf)[0]
    if prediction==1:
        result='cyberbullying'
    else:
        result='safe'

    cursor.execute("INSERT INTO comments(comment,prediction) VALUES (?,?)",(comment,result))
    conn.commit()
    return result


app=gr.Interface(fn=predict_comment,
             inputs=gr.Textbox(lines=4,placeholder='Enter a social media comment....'),
                 outputs="text",title='Cyberbullying Detection System',description='Detect whther a comment is cyberbullying or safe')


app.launch()







