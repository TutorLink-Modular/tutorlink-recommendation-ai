import os
from pymongo import MongoClient
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)

# ✨ Obtener la colección según categoría
def get_collection_by_category(category):
    db_name = mongo_uri.split("/")[-1].split("?")[0]
    db = client[db_name]

    if category == "disciplinar":
        return db["topicsDisciplinary"]
    elif category == "orientacional":
        return db["topicsOrientation"]
    else:
        raise ValueError("Categoría inválida")

def recommend(title, category="disciplinar", n_recommendations=3):
    # 🔍 Obtener colección desde MongoDB según categoría
    collection = get_collection_by_category(category)

    # 🔹 Incluimos también el campo "image"
    data = pd.DataFrame(list(collection.find(
        {}, {"title": 1, "description": 1, "text": 1, "image": 1, "_id": 1}
    )))

    if data.empty:
        raise ValueError("No hay datos en la colección.")

    if title not in data["title"].values:
        return []

    # 🔠 Vectorizamos el campo de texto
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(data["text"])

    idx = data.index[data["title"] == title][0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    similar_indices = cosine_sim.argsort()[-(n_recommendations + 1):-1][::-1]

    # 🔹 Incluimos el campo "image" en la salida
    recommended_topics = data.iloc[similar_indices][["_id", "title", "description", "image"]].to_dict(orient="records")

    return recommended_topics
