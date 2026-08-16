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

# Allow requests from your Vercel frontend
# We are keeping this open for now so that CORS does not hide
# the actual backend error while we debug the deployment.
CORS(app)

# Folder where uploaded files, progress files and quiz files are stored
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------------------------------------------------
# Absolute path to this file's directory, and to gemini3.py.
#
# The previous version called subprocess.run(["gemini3.py", ...])
# using a bare relative path. That only works if the process's
# current working directory happens to be this file's directory.
# Depending on how the platform (Render, gunicorn, etc.) starts
# the app, that is not guaranteed. Using an absolute path removes
# that ambiguity entirely.
# --------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GEMINI_SCRIPT_PATH = os.path.join(BASE_DIR, "gemini3.py")


# ============================================================
# HOME / HEALTH ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "PrepAI backend is running"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok"
    })


# ============================================================
# URL TEXT EXTRACTION
# ============================================================

def extract_text_from_url(url):
    """
    Extract readable text from a webpage URL.

    Special handling is included for Wikipedia.
    """

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Special handling for Wikipedia
        content = soup.find(
            "div",
            {"id": "mw-content-text"}
        )

        if content:
            return content.get_text(
                separator="\n",
                strip=True
            )

        # Fallback: extract paragraphs
        paragraphs = soup.find_all("p")

        text = "\n".join(
            p.get_text(strip=True)
            for p in paragraphs
        )

        if text.strip():
            return text

        # Final fallback
        return soup.get_text(
            separator="\n",
            strip=True
        )

    except Exception as e:
        print("❌ Error extracting URL:", str(e))

        return f"Error extracting URL: {e}"


# ============================================================
# QUIZ GENERATION
# ============================================================

def run_quiz_generation(
    file_path,
    file_id,
    num_questions,
    question_types
):
    """
    Run gemini3.py in the background.

    The generated quiz is saved as:

        uploads/<file_id>_quiz.json

    Progress is saved as:

        uploads/<file_id>_progress.json
    """

    quiz_json = os.path.join(
        UPLOAD_FOLDER,
        f"{file_id}_quiz.json"
    )

    progress_json = os.path.join(
        UPLOAD_FOLDER,
        f"{file_id}_progress.json"
    )

    # --------------------------------------------------------
    # Mark generation as started
    # --------------------------------------------------------

    with open(progress_json, "w") as f:
        json.dump(
            {
                "status": "running",
                "progress": 10
            },
            f
        )

    try:

        # ----------------------------------------------------
        # Prepare question types
        # ----------------------------------------------------

        types_str = ",".join(question_types)

        print()
        print("========================================")
        print("STARTING QUIZ GENERATION")
        print("========================================")
        print("File:", file_path)
        print("File ID:", file_id)
        print("Number of questions:", num_questions)
        print("Question types:", types_str)
        print("Quiz output:", quiz_json)
        print("Gemini script:", GEMINI_SCRIPT_PATH)
        print("========================================")
        print(flush=True)

        # ----------------------------------------------------
        # Update progress
        # ----------------------------------------------------

        with open(progress_json, "w") as f:
            json.dump(
                {
                    "status": "running",
                    "progress": 30
                },
                f
            )

        # ----------------------------------------------------
        # Run gemini3.py
        #
        # Use the absolute path (GEMINI_SCRIPT_PATH) so this
        # works regardless of the process's working directory.
        # ----------------------------------------------------

        result = subprocess.run(
            [
                sys.executable,
                GEMINI_SCRIPT_PATH,

                "--input",
                file_path,

                "--out",
                quiz_json,

                "--max_questions",
                str(num_questions),

                "--types",
                types_str
            ],

            capture_output=True,
            text=True,

            # Maximum 10 minutes
            timeout=600
        )

        # ----------------------------------------------------
        # VERY IMPORTANT:
        # Print output from gemini3.py to Render logs
        # ----------------------------------------------------

        print()
        print("========================================")
        print("GEMINI STDOUT")
        print("========================================")
        print(result.stdout)

        print()
        print("========================================")
        print("GEMINI STDERR")
        print("========================================")
        print(result.stderr)

        print()
        print("========================================")
        print("GEMINI RETURN CODE")
        print("========================================")
        print(result.returncode, flush=True)

        print()

        # ----------------------------------------------------
        # Check whether generation succeeded
        # ----------------------------------------------------

        if (
            result.returncode == 0
            and os.path.exists(quiz_json)
        ):

            with open(progress_json, "w") as f:
                json.dump(
                    {
                        "status": "done",
                        "progress": 100
                    },
                    f
                )

            print(
                f"✅ Quiz generation complete for {file_id}",
                flush=True
            )

        else:

            error_message = (
                "Quiz generation failed. "
                f"Return code: {result.returncode}. "
                f"STDERR: {result.stderr}"
            )

            print()
            print("========================================")
            print("QUIZ GENERATION FAILED")
            print("========================================")
            print(error_message)
            print("========================================")
            print(flush=True)

            with open(progress_json, "w") as f:
                json.dump(
                    {
                        "status": "error",
                        "error": error_message
                    },
                    f
                )

    # --------------------------------------------------------
    # Timeout
    # --------------------------------------------------------

    except subprocess.TimeoutExpired:

        print(
            "❌ Quiz generation timed out after 10 minutes",
            flush=True
        )

        with open(progress_json, "w") as f:
            json.dump(
                {
                    "status": "error",
                    "error": (
                        "Quiz generation timed out "
                        "after 10 minutes"
                    )
                },
                f
            )

    # --------------------------------------------------------
    # Any other error
    #
    # This is the important safety net: if anything above
    # raises (including things like the file not existing,
    # a permissions issue, etc.) we still write an "error"
    # status instead of leaving progress.json stuck on
    # "running" forever.
    # --------------------------------------------------------

    except Exception as e:

        print(
            "❌ Unexpected error during quiz generation:",
            str(e),
            flush=True
        )

        with open(progress_json, "w") as f:
            json.dump(
                {
                    "status": "error",
                    "error": str(e)
                },
                f
            )


# ============================================================
# UPLOAD ENDPOINT
# ============================================================

@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Accept either:

    1. PDF / DOCX / TXT file
    OR
    2. URL

    Then start quiz generation in the background.
    """

    uploaded_file = request.files.get("file")

    url = request.form.get(
        "url",
        ""
    ).strip()

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not uploaded_file and not url:

        return jsonify(
            {
                "error": "No file or URL provided"
            }
        ), 400

    # --------------------------------------------------------
    # Read quiz parameters
    # --------------------------------------------------------

    try:

        num_questions = int(
            request.form.get(
                "num_questions",
                20
            )
        )

        question_types_str = request.form.get(
            "question_types",
            "mcq,short,fillblank,tf"
        )

        question_types = [
            t.strip()
            for t in question_types_str.split(",")
            if t.strip()
        ]

        # Supported question types
        valid_types = [
            "mcq",
            "short",
            "fillblank",
            "tf"
        ]

        question_types = [
            qt
            for qt in question_types
            if qt in valid_types
        ]

        if not question_types:

            return jsonify(
                {
                    "error": (
                        "No valid question types selected"
                    )
                }
            ), 400

        if num_questions < 1 or num_questions > 100:

            return jsonify(
                {
                    "error": (
                        "Number of questions "
                        "must be between 1 and 100"
                    )
                }
            ), 400

    except ValueError:

        return jsonify(
            {
                "error": "Invalid parameters"
            }
        ), 400

    # --------------------------------------------------------
    # Generate unique ID
    # --------------------------------------------------------

    file_id = str(uuid.uuid4())

    # --------------------------------------------------------
    # CASE 1: FILE UPLOAD
    # --------------------------------------------------------

    if uploaded_file:

        file_ext = os.path.splitext(
            uploaded_file.filename
        )[1].lower()

        if file_ext not in [
            ".pdf",
            ".docx",
            ".txt"
        ]:

            return jsonify(
                {
                    "error": (
                        "Only PDF, DOCX, and TXT "
                        "files are supported"
                    )
                }
            ), 400

        file_path = os.path.join(
            UPLOAD_FOLDER,
            f"{file_id}{file_ext}"
        )

        uploaded_file.save(file_path)

        print(
            f"📄 Saved uploaded file at: {file_path}",
            flush=True
        )

    # --------------------------------------------------------
    # CASE 2: URL
    # --------------------------------------------------------

    elif url:

        file_path = os.path.join(
            UPLOAD_FOLDER,
            f"{file_id}.txt"
        )

        extracted_text = extract_text_from_url(url)

        with open(
            file_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(extracted_text)

        print(
            f"🌐 URL extracted and saved at: {file_path}",
            flush=True
        )

    # --------------------------------------------------------
    # Start quiz generation
    # --------------------------------------------------------

    print()
    print("========================================")
    print("QUIZ REQUEST RECEIVED")
    print("========================================")
    print("File ID:", file_id)
    print("Questions:", num_questions)
    print("Question types:", question_types)
    print("========================================")
    print(flush=True)

    thread = threading.Thread(
        target=run_quiz_generation,

        args=(
            file_path,
            file_id,
            num_questions,
            question_types
        )
    )

    thread.daemon = True

    thread.start()

    # --------------------------------------------------------
    # Immediately return file ID
    # --------------------------------------------------------

    return jsonify(
        {
            "file_id": file_id,

            "message": (
                "Input received, "
                "quiz generation started"
            ),

            "num_questions": num_questions,

            "question_types": question_types
        }
    )


# ============================================================
# QUIZ STATUS / RESULT ENDPOINT
# ============================================================

@app.route(
    "/quiz/<file_id>",
    methods=["GET"]
)
def get_quiz(file_id):
    """
    Return the current status of quiz generation.

    Possible responses:

        pending
        running
        done
        error
    """

    progress_json = os.path.join(
        UPLOAD_FOLDER,
        f"{file_id}_progress.json"
    )

    quiz_json = os.path.join(
        UPLOAD_FOLDER,
        f"{file_id}_quiz.json"
    )

    # --------------------------------------------------------
    # Progress file does not exist yet
    # --------------------------------------------------------

    if not os.path.exists(progress_json):

        return jsonify(
            {
                "status": "pending",
                "progress": 0
            }
        ), 202

    # --------------------------------------------------------
    # Read progress
    # --------------------------------------------------------

    try:

        with open(
            progress_json,
            "r"
        ) as f:

            progress = json.load(f)

    except Exception as e:

        print(
            "❌ Error reading progress file:",
            str(e)
        )

        return jsonify(
            {
                "status": "error",
                "error": (
                    "Could not read quiz progress"
                )
            }
        ), 500

    # --------------------------------------------------------
    # Quiz finished
    # --------------------------------------------------------

    if (
        progress.get("status") == "done"
        and os.path.exists(quiz_json)
    ):

        try:

            with open(
                quiz_json,
                "r",
                encoding="utf-8"
            ) as f:

                quiz = json.load(f)

            return jsonify(
                {
                    "status": "done",
                    "quiz": quiz
                }
            )

        except Exception as e:

            print(
                "❌ Error reading quiz JSON:",
                str(e)
            )

            return jsonify(
                {
                    "status": "error",
                    "error": (
                        "Could not read generated quiz"
                    )
                }
            ), 500

    # --------------------------------------------------------
    # Quiz still generating
    # --------------------------------------------------------

    elif progress.get("status") == "running":

        return jsonify(
            {
                "status": "running",

                "progress": progress.get(
                    "progress",
                    0
                )
            }
        ), 202

    # --------------------------------------------------------
    # Quiz generation failed
    # --------------------------------------------------------

    else:

        return jsonify(
            {
                "status": "error",

                "error": progress.get(
                    "error",
                    "Unknown error"
                )
            }
        ), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print(
        "🚀 Starting Flask server..."
    )

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )
