from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import uuid
import os
import json
import threading

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def run_quiz_generation(file_path, file_id):
    """Run rag_quiz_generator_gemini_opt.py in background and update progress."""
    quiz_json = os.path.join(UPLOAD_FOLDER, f"{file_id}_quiz.json")
    progress_json = os.path.join(UPLOAD_FOLDER, f"{file_id}_progress.json")
    
    # Initialize progress
    with open(progress_json, "w") as f:
        json.dump({"status": "running", "progress": 10}, f)
    
    try:
        # Update progress to 30% when starting generation
        with open(progress_json, "w") as f:
            json.dump({"status": "running", "progress": 30}, f)
        
        # Call the RAG quiz generator script (your gemini3.py)
        result = subprocess.run(
            [
                "python", 
                "gemini3.py",  # Using your actual filename
                "--input", file_path, 
                "--out", quiz_json,
                "--max_questions", "20"
            ],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        # Check if quiz file was created successfully
        if os.path.exists(quiz_json):
            with open(progress_json, "w") as f:
                json.dump({"status": "done", "progress": 100}, f)
            print(f"✅ Quiz generation complete for {file_id}")
        else:
            raise Exception(f"Quiz file not created. stderr: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        with open(progress_json, "w") as f:
            json.dump({"status": "error", "error": "Quiz generation timed out"}, f)
    except Exception as e:
        with open(progress_json, "w") as f:
            json.dump({"status": "error", "error": str(e)}, f)
        print(f"❌ Error generating quiz: {e}")


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload and start quiz generation."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Generate unique file ID
    file_id = str(uuid.uuid4())
    
    # Get file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ['.pdf', '.docx', '.txt']:
        return jsonify({"error": "Only PDF, DOCX, and TXT files are supported"}), 400
    
    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}{file_ext}")
    file.save(file_path)
    print(f"📄 Saved file at: {file_path}")

    # Start background thread for quiz generation
    thread = threading.Thread(target=run_quiz_generation, args=(file_path, file_id))
    thread.daemon = True
    thread.start()

    return jsonify({
        "file_id": file_id, 
        "message": "File uploaded, quiz generation started"
    })


@app.route("/quiz/<file_id>", methods=["GET"])
def get_quiz(file_id):
    """Get quiz status or data."""
    progress_json = os.path.join(UPLOAD_FOLDER, f"{file_id}_progress.json")
    quiz_json = os.path.join(UPLOAD_FOLDER, f"{file_id}_quiz.json")

    # Check if progress file exists
    if not os.path.exists(progress_json):
        return jsonify({"status": "pending", "progress": 0}), 202

    # Read progress
    with open(progress_json, "r") as f:
        progress = json.load(f)

    # If done, return quiz data
    if progress.get("status") == "done" and os.path.exists(quiz_json):
        with open(quiz_json, "r", encoding="utf-8") as f:
            quiz = json.load(f)
        return jsonify({"status": "done", "quiz": quiz})
    
    # If still running, return progress
    elif progress.get("status") == "running":
        return jsonify({
            "status": "running", 
            "progress": progress.get("progress", 0)
        }), 202
    
    # If error occurred
    else:
        return jsonify({
            "status": "error", 
            "error": progress.get("error", "Unknown error")
        }), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("🚀 Starting Flask server on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)