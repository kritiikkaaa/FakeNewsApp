from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
with open("fake_news_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    news_text = ""
    submitted = False

    if request.method == "POST":
        submitted = True
        news_text = request.form.get("news_text", "").strip()

        if news_text:
            transformed_input = vectorizer.transform([news_text])
            prediction = model.predict(transformed_input)[0]

    return render_template("index.html", prediction=prediction, news_text=news_text, submitted=submitted)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)