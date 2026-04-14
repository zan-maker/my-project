"""
embedding-server.py — Local embedding server for SemantiQ

Runs sentence-transformers/all-MiniLM-L6-v2 and exposes a lightweight
HTTP API for generating text embeddings. Used by vector-store.ts.

Usage:
    pip install sentence-transformers flask flask-cors
    python embedding-server.py

Endpoints:
    POST /api/embed        {"text": "..."}              → {"embedding": [0.01, ...]}
    POST /api/embed-batch  {"texts": ["...", "..."]}    → {"embeddings": [[...], [...]]}
    GET  /api/health        → {"status": "ok", "model": "..."}
"""

import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
model = None

def get_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading model: {MODEL_NAME}...")
        model = SentenceTransformer(MODEL_NAME)
        print("Model loaded.")
    return model

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "loaded": model is not None,
    })

@app.route("/api/embed", methods=["POST"])
def embed():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "text is required"}), 400

    m = get_model()
    embedding = m.encode(text).tolist()
    return jsonify({"embedding": embedding})

@app.route("/api/embed-batch", methods=["POST"])
def embed_batch():
    data = request.get_json()
    texts = data.get("texts", [])
    if not texts:
        return jsonify({"error": "texts is required"}), 400

    m = get_model()
    embeddings = m.encode(texts).tolist()
    return jsonify({"embeddings": embeddings})

if __name__ == "__main__":
    print(f"Starting embedding server on port 11434...")
    print(f"Model: {MODEL_NAME}")
    app.run(host="0.0.0.0", port=11434, debug=False)
