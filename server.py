from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import uuid
import os
import json
import threading
import sys
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def extract_text_from_url(url):
    """Extract clean text from a webpage URL with proper headers and content filtering."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Special handling for Wikipedia
        content = soup.find("div", {"id": "mw-content-text"})
        if content:
            return content.get_text(separator="\n", strip=True)

        # Fallback: extract only meaningful text
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text(strip=True) for p in paragraphs)

        if text.strip():
            return text

        # Extra fallback
        return soup.get_text(separator="\n", strip=True)

    except Exception as e:
        return f"Error extracting URL: {e}"



def run_quiz_generation(file_path, file_id, num_questions, question_types):
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
        
        # Prepare question types string
        types_str = ",".join(question_types)
        
        # Call the RAG quiz generator script (gemini3.py)
        result = subprocess.run(
            [
                sys.executable, 
                "gemini3.py",
                "--input", file_path, 
                "--out", quiz_json,
                "--max_questions", str(num_questions),
                "--types", types_str
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
    """Handle file upload OR URL input and start quiz generation."""

    uploaded_file = request.files.get("file")
    url = request.form.get("url", "").strip()

    # If neither file nor URL provided
    if not uploaded_file and not url:
        return jsonify({"error": "No file or URL provided"}), 400

    # Get quiz parameters
    try:
        num_questions = int(request.form.get("num_questions", 20))
        question_types_str = request.form.get("question_types", "mcq,short,fillblank,tf")
        question_types = [t.strip() for t in question_types_str.split(",") if t.strip()]
        
        valid_types = ["mcq", "short", "fillblank", "tf"]
        question_types = [qt for qt in question_types if qt in valid_types]
        
        if not question_types:
            return jsonify({"error": "No valid question types selected"}), 400
            
        if num_questions < 1 or num_questions > 100:
            return jsonify({"error": "Number of questions must be between 1 and 100"}), 400
            
    except ValueError:
        return jsonify({"error": "Invalid parameters"}), 400

    # Generate unique file ID
    file_id = str(uuid.uuid4())

    # CASE 1: IF FILE IS UPLOADED
    if uploaded_file:
        file_ext = os.path.splitext(uploaded_file.filename)[1].lower()
        if file_ext not in ['.pdf', '.docx', '.txt']:
            return jsonify({"error": "Only PDF, DOCX, and TXT files are supported"}), 400
        
        file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}{file_ext}")
        uploaded_file.save(file_path)
        print(f"📄 Saved file at: {file_path}")

    # CASE 2: IF URL IS PROVIDED
    elif url:
        file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.txt")
        extracted_text = extract_text_from_url(url)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)

        print(f"🌐 URL extracted and saved at: {file_path}")

    print(f"📊 Generating {num_questions} questions of types: {question_types}")

    # Start background thread
    thread = threading.Thread(
        target=run_quiz_generation, 
        args=(file_path, file_id, num_questions, question_types)
    )
    thread.daemon = True
    thread.start()

    return jsonify({
        "file_id": file_id, 
        "message": "Input received, quiz generation started",
        "num_questions": num_questions,
        "question_types": question_types
    })


@app.route("/quiz/<file_id>", methods=["GET"])
def get_quiz(file_id):
    """Get quiz status or data."""
    progress_json = os.path.join(UPLOAD_FOLDER, f"{file_id}_progress.json")
    quiz_json = os.path.join(UPLOAD_FOLDER, f"{file_id}_quiz.json")

    if not os.path.exists(progress_json):
        return jsonify({"status": "pending", "progress": 0}), 202

    with open(progress_json, "r") as f:
        progress = json.load(f)

    if progress.get("status") == "done" and os.path.exists(quiz_json):
        with open(quiz_json, "r", encoding="utf-8") as f:
            quiz = json.load(f)
        return jsonify({"status": "done", "quiz": quiz})
    
    elif progress.get("status") == "running":
        return jsonify({"status": "running", "progress": progress.get("progress", 0)}), 202

    else:
        return jsonify({"status": "error", "error": progress.get("error", "Unknown error")}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("🚀 Starting Flask server on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
