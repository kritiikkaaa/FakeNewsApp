import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle

# Sample dataset
data = {
    "text": [
        "Donald Trump says vaccines cause autism.",
        "NASA confirms water on the moon.",
        "COVID-19 is a hoax spread by governments.",
        "Scientists discover new species in Amazon rainforest.",
        "5G towers spread coronavirus, experts say.",
        "Apple releases new iPhone with improved camera.",
        "Bill Gates created the virus to control population.",
        "WHO approves new malaria vaccine for global use."
    ],
    "label": ["FAKE", "FAKE", "FAKE", "REAL", "FAKE", "REAL", "FAKE", "REAL"]
}

df = pd.DataFrame(data)
df.to_csv("news.csv", index=False)

# Train model
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.25, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, y_train)

# Save model and vectorizer
with open("fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f) 