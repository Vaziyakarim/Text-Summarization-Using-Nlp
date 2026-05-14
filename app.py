from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import os

# Download required NLTK data
nltk.download("punkt")

# Initialize Flask app
app = Flask(__name__, template_folder=os.path.abspath("templates"))
CORS(app)  # Fix CORS issues

# Load Abstractive Summarization Model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Extractive Summarization Function
def extractive_summary(text, num_sentences=3):
    sentences = sent_tokenize(text)
    return " ".join(sentences[:num_sentences])

# Abstractive Summarization Function
def abstractive_summary(text, max_length=150):
    if len(text.split()) < 50:  # Ensure text is long enough
        return "Text too short for abstractive summarization."
    
    summary = summarizer(text, max_length=max_length, min_length=50, do_sample=False)
    return summary[0]['summary_text']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Invalid request format"}), 400

        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Input text is empty"}), 400

        num_sentences = int(data.get("num_sentences", 3))
        max_length = int(data.get("max_length", 150))

        extractive = extractive_summary(text, num_sentences)
        abstractive = abstractive_summary(text, max_length)

        return jsonify({
            "extractive_summary": extractive,
            "abstractive_summary": abstractive
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
