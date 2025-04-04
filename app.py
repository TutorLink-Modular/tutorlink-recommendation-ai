from flask import Flask, jsonify, request
from flask_cors import CORS
from recommendation_model import recommend

app = Flask(__name__)
CORS(app)

@app.route("/recommend", methods=["GET"])
def get_recommendations():
    title = request.args.get("title")
    category = request.args.get("category")  # Nuevo parámetro

    if not title or not category:
        return jsonify({"error": "Se requieren los parámetros 'title' y 'category'"}), 400

    try:
        recommendations = recommend(title, category)
        return jsonify(recommendations), 200
    except Exception as e:
        print(f"Error en recomendación:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
