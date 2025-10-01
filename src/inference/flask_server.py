import os
from flask import Flask, request, render_template, jsonify
from process_audio import process_audio_file

app = Flask(__name__, template_folder="templates")

# Path to your trained Sprint model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "saved_models", "sprint_model_v5_best.pth")

@app.route("/")
def index():
    return render_template("index.html")  # simple frontend with mic recording

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save temp file
    upload_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(upload_path)

    try:
        result = process_audio_file(upload_path, MODEL_PATH)

        return jsonify({
            "transcript": result["transcript"],
            "sentiment": result["sentiment"],
            "emotion": result["emotion"],
            "attention": {
                "audio": result["attention"]["audio"],  # Remove the * 100
                "text": result["attention"]["text"]     # Remove the * 100
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(upload_path):
            os.remove(upload_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
