from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model and vectorizer
with open(os.path.join(BASE_DIR, "fake_news_model.pkl"), "rb") as model_file:
    model = pickle.load(model_file)

with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as vec_file:
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

    return render_template(
        "index.html",
        prediction=prediction,
        news_text=news_text,
        submitted=submitted
    )


if __name__ == "__main__":
    app.run(debug=True)