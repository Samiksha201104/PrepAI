#!/usr/bin/env python3
"""
RAG Quiz Generator using Gemini + TF-IDF retrieval

Supports:
    - PDF
    - DOCX
    - TXT
    - URL

Question types:
    - mcq
    - short
    - fillblank
    - tf

Designed to be called by server.py.

NOTE ON THIS VERSION:
Retrieval previously used sentence-transformers + faiss (i.e. a full
PyTorch install). On memory-constrained hosts (e.g. Render free/starter
tier, ~512MB RAM), importing torch + sentence-transformers alone can
exceed the memory limit and get the process OOM-killed mid-request,
which is why quiz generation would silently hang at "running" forever.

This version uses scikit-learn's TF-IDF + cosine similarity instead.
It's a few MB instead of 1GB+, starts instantly (no model download),
and is more than good enough for retrieving relevant chunks from a
single document.
"""

import os
import re
import json
import time
import random
import argparse

from pathlib import Path
from typing import List, Tuple, Optional

import requests
import pdfplumber
from docx import Document
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import google.generativeai as genai


# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash-lite"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CHUNK_WORDS = int(
    os.getenv("CHUNK_WORDS", "600")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "120")
)

MAX_GEN_RETRIES = 3

BACKOFF_BASE = 1.5

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# GEMINI SETUP
# ============================================================

if not GEMINI_API_KEY:

    print(
        "WARNING: GEMINI_API_KEY is not set.",
        flush=True
    )

else:

    genai.configure(
        api_key=GEMINI_API_KEY
    )


# ============================================================
# RETRY HELPER
# ============================================================

def retry_with_backoff(
    fn,
    retries=MAX_GEN_RETRIES,
    base=BACKOFF_BASE
):
    """
    Retry a function if it fails.

    Uses exponential backoff.
    """

    for attempt in range(1, retries + 1):

        try:

            return fn()

        except Exception as e:

            if attempt == retries:
                raise

            sleep_time = base ** attempt

            print(
                f"[retry] Attempt {attempt} failed: {e}. "
                f"Sleeping {sleep_time:.1f}s",
                flush=True
            )

            time.sleep(sleep_time)


# ============================================================
# GEMINI JSON CLEANING
# ============================================================

def clean_gemini_json(
    raw_text: str
) -> Optional[dict]:

    """
    Convert Gemini output into a Python dictionary.

    Handles:
        ```json
        {...}
        ```

    and cases where Gemini adds extra text.
    """

    if not raw_text:
        return None

    cleaned = raw_text.strip()

    # --------------------------------------------------------
    # Remove markdown code fences
    # --------------------------------------------------------

    cleaned = re.sub(
        r"```json\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"```\s*",
        "",
        cleaned
    )

    cleaned = cleaned.strip()

    # --------------------------------------------------------
    # First attempt: parse entire response
    # --------------------------------------------------------

    try:

        return json.loads(cleaned)

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # Second attempt: extract JSON object
    # --------------------------------------------------------

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:

        candidate = cleaned[
            start:end + 1
        ]

        try:

            return json.loads(candidate)

        except json.JSONDecodeError as e:

            print(
                f"[warn] JSON parse error: {e}",
                flush=True
            )

    print(
        f"[warn] Could not parse Gemini JSON: "
        f"{raw_text[:300]}",
        flush=True
    )

    return None


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(
    path: str
) -> str:

    parts = []

    with pdfplumber.open(path) as pdf:

        for page in pdf.pages:

            text = page.extract_text()

            if text:

                parts.append(text)

    return "\n\n".join(parts)


def extract_text_from_docx(
    path: str
) -> str:

    doc = Document(path)

    paragraphs = []

    for paragraph in doc.paragraphs:

        text = paragraph.text.strip()

        if text:

            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def extract_text_from_txt(
    path: str
) -> str:

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        return f.read()


def extract_text_from_url(
    url: str
) -> str:

    def _get():

        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=20
        )

        response.raise_for_status()

        return response.text

    html = retry_with_backoff(_get)

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    # --------------------------------------------------------
    # Remove unwanted elements
    # --------------------------------------------------------

    for element in soup(
        [
            "script",
            "style",
            "noscript",
            "header",
            "footer",
            "nav"
        ]
    ):

        element.decompose()

    # --------------------------------------------------------
    # Prefer article/main content
    # --------------------------------------------------------

    main_content = (
        soup.find("article")
        or soup.find("main")
    )

    if main_content:

        paragraphs = main_content.find_all("p")

    else:

        paragraphs = soup.find_all("p")

    text_parts = []

    for paragraph in paragraphs:

        text = paragraph.get_text(
            " ",
            strip=True
        )

        if text:

            text_parts.append(text)

    text = "\n\n".join(text_parts)

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if not text.strip():

        text = soup.get_text(
            separator="\n",
            strip=True
        )

    return text


def extract_text(
    path_or_url: str
) -> str:

    # --------------------------------------------------------
    # URL
    # --------------------------------------------------------

    if (
        path_or_url.startswith("http://")
        or path_or_url.startswith("https://")
    ):

        return extract_text_from_url(
            path_or_url
        )

    # --------------------------------------------------------
    # File
    # --------------------------------------------------------

    path = Path(path_or_url)

    if not path.exists():

        raise FileNotFoundError(
            f"Input file not found: {path_or_url}"
        )

    extension = path.suffix.lower()

    if extension == ".pdf":

        return extract_text_from_pdf(
            path_or_url
        )

    elif extension == ".docx":

        return extract_text_from_docx(
            path_or_url
        )

    elif extension == ".txt":

        return extract_text_from_txt(
            path_or_url
        )

    else:

        raise ValueError(
            f"Unsupported input format: {extension}"
        )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(
    text: str
) -> str:

    if not text:

        return ""

    # Normalize newlines
    text = re.sub(
        r"\r\n|\r",
        "\n",
        text
    )

    # Remove page numbers
    text = re.sub(
        r"(?im)^(page|pg)\s*\d+\b.*$",
        "",
        text
    )

    # Remove confidential/copyright lines
    text = re.sub(
        r"(?im)^(confidential|copyright).*$",
        "",
        text
    )

    # Remove URLs
    text = re.sub(
        r"http\S+|www\.\S+",
        "",
        text
    )

    # Remove citation numbers like [1], [23]
    text = re.sub(
        r"\[[0-9]+\]",
        "",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(
    text: str,
    chunk_words: int = CHUNK_WORDS,
    overlap: int = CHUNK_OVERLAP
) -> List[str]:

    words = text.split()

    if not words:

        return []

    # Prevent invalid configuration
    if overlap >= chunk_words:

        overlap = chunk_words // 5

    step = chunk_words - overlap

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_words

        chunk = words[start:end]

        if chunk:

            chunks.append(
                " ".join(chunk)
            )

        start += step

    return chunks


# ============================================================
# TF-IDF VECTOR STORE (replaces FAISS + sentence-transformers)
# ============================================================

class TfidfStore:
    """
    Lightweight retrieval store using TF-IDF + cosine similarity.

    Deliberately avoids sentence-transformers/faiss (which pull in a
    full PyTorch install and can easily exceed memory limits on
    small hosting instances). scikit-learn's TF-IDF is a few MB,
    has no model download step, and is fast enough for retrieval
    over a single document's worth of chunks.
    """

    def __init__(self):

        self.vectorizer = None

        self.matrix = None

        self.chunks: List[str] = []


    def build(
        self,
        chunks: List[str]
    ):

        if not chunks:

            self.vectorizer = None

            self.matrix = None

            self.chunks = []

            return

        print(
            f"[info] Building TF-IDF index for "
            f"{len(chunks)} chunks...",
            flush=True
        )

        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        self.matrix = self.vectorizer.fit_transform(
            chunks
        )

        self.chunks = chunks

        print(
            "[info] TF-IDF index built",
            flush=True
        )


    def search(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Tuple[int, float]]:

        if self.vectorizer is None or not self.chunks:

            return []

        # Do not request more results than available chunks
        top_k = min(
            top_k,
            len(self.chunks)
        )

        query_vec = self.vectorizer.transform(
            [query]
        )

        scores = cosine_similarity(
            query_vec,
            self.matrix
        )[0]

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for idx in ranked_indices:

            score = float(scores[idx])

            if score <= 0:
                continue

            results.append(
                (int(idx), score)
            )

        return results


    def get_text(
        self,
        idx: int
    ) -> str:

        return self.chunks[idx]


# ============================================================
# GEMINI API
# ============================================================

def call_gemini(
    prompt: str,
    model_name: str = GEMINI_MODEL,
    temperature: float = 0.0
) -> str:

    if not GEMINI_API_KEY:

        raise RuntimeError(
            "GEMINI_API_KEY environment variable "
            "is not set."
        )

    model_name = (
        model_name
        .replace("models/", "")
        .strip()
    )

    def _call():

        model = genai.GenerativeModel(
            model_name
        )

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature
            }
        )

        if getattr(
            response,
            "text",
            None
        ):

            return response.text

        return ""

    return retry_with_backoff(
        _call
    )


# ============================================================
# PROMPTS
# ============================================================

PROMPT_SUMMARY = """
You are a study assistant.

Summarize ONLY the information contained in the provided context.

Create 3-5 concise bullet points.

Each bullet should contain approximately 10-25 words.

Do not introduce information that is not present in the context.

CONTEXT:
{context}

OUTPUT:
"""


PROMPT_MCQ = """
You are an educational quiz generator.

Create ONE multiple-choice question based ONLY on the provided context.

Requirements:

1. The question must test an important concept from the context.
2. Create exactly four options.
3. Options must be labeled A, B, C, and D.
4. Exactly ONE option must be correct.
5. The other options must be plausible but incorrect.
6. Do not use information that is not present in the context.
7. Do not make the correct answer obvious because of wording or length.
8. Return ONLY valid JSON.
9. Do not use markdown.
10. Do not add any text before or after the JSON.

Return exactly this structure:

{
  "question": "question text",
  "choices": {
    "A": "option A",
    "B": "option B",
    "C": "option C",
    "D": "option D"
  },
  "answer_letter": "A",
  "answer_text": "correct option text",
  "explanation": "one sentence explaining why the answer is correct"
}

CONTEXT:
{context}

JSON:
"""


PROMPT_SHORT = """
You are an educational quiz generator.

Create ONE short-answer question based ONLY on the provided context.

Requirements:

1. Test an important concept.
2. The answer should be directly supported by the context.
3. Keep the expected answer reasonably short.
4. Do not introduce outside information.
5. Return ONLY valid JSON.
6. Do not use markdown.
7. Do not add any text before or after the JSON.

Return exactly:

{
  "question": "question text",
  "answer": "correct answer",
  "explanation": "one sentence explanation"
}

CONTEXT:
{context}

JSON:
"""


PROMPT_FILL = """
You are an educational quiz generator.

Create ONE fill-in-the-blank question based ONLY on the provided context.

Requirements:

1. The question must contain exactly one blank represented by _____.
2. The blank should correspond to an important word or short phrase.
3. The answer must be directly supported by the context.
4. Do not introduce outside information.
5. Return ONLY valid JSON.
6. Do not use markdown.
7. Do not add any text before or after the JSON.

Return exactly:

{
  "question": "question containing _____",
  "answer": "word or phrase that belongs in the blank",
  "explanation": "one sentence explanation"
}

CONTEXT:
{context}

JSON:
"""


PROMPT_TF = """
You are an educational quiz generator.

Create ONE True/False question based ONLY on the provided context.

Requirements:

1. Create a clear factual statement.
2. The statement must be either definitely true or definitely false according to the context.
3. Do not use ambiguous wording.
4. Do not introduce outside information.
5. The answer must be exactly "True" or "False".
6. Return ONLY valid JSON.
7. Do not use markdown.
8. Do not add any text before or after the JSON.

Return exactly:

{
  "question": "statement",
  "answer": "True",
  "explanation": "one sentence explanation"
}

CONTEXT:
{context}

JSON:
"""


# ============================================================
# QUESTION VALIDATION
# ============================================================

def validate_question(
    item: dict,
    qtype: str
) -> bool:

    if not isinstance(
        item,
        dict
    ):

        return False

    # --------------------------------------------------------
    # MCQ
    # --------------------------------------------------------

    if qtype == "mcq":

        required = [
            "question",
            "choices",
            "answer_letter"
        ]

        if not all(
            key in item
            for key in required
        ):

            return False

        choices = item.get(
            "choices"
        )

        if not isinstance(
            choices,
            dict
        ):

            return False

        if set(
            choices.keys()
        ) != {"A", "B", "C", "D"}:

            return False

        if item.get(
            "answer_letter"
        ) not in ["A", "B", "C", "D"]:

            return False

        return bool(
            item.get("question")
        )


    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    if qtype == "short":

        return bool(
            item.get("question")
            and item.get("answer")
        )


    # --------------------------------------------------------
    # FILL BLANK
    # --------------------------------------------------------

    if qtype == "fillblank":

        question = item.get(
            "question",
            ""
        )

        return (
            "_____" in question
            and bool(item.get("answer"))
        )


    # --------------------------------------------------------
    # TRUE / FALSE
    # --------------------------------------------------------

    if qtype == "tf":

        return (
            bool(item.get("question"))
            and item.get("answer")
            in ["True", "False"]
        )

    return False


# ============================================================
# QUESTION GENERATION
# ============================================================

def generate_one_question(
    context: str,
    qtype: str
) -> dict:

    context = context.strip()

    if not context:

        return {
            "type": qtype,
            "error": "empty context"
        }

    # --------------------------------------------------------
    # Select prompt
    # --------------------------------------------------------

    if qtype == "mcq":

        prompt = PROMPT_MCQ

    elif qtype == "short":

        prompt = PROMPT_SHORT

    elif qtype == "fillblank":

        prompt = PROMPT_FILL

    elif qtype == "tf":

        prompt = PROMPT_TF

    else:

        raise ValueError(
            f"Unsupported question type: {qtype}"
        )

    # --------------------------------------------------------
    # Call Gemini
    # --------------------------------------------------------

    raw = call_gemini(
        prompt.replace(
            "{context}", context
        )
    )

    parsed = clean_gemini_json(
        raw
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not validate_question(
        parsed,
        qtype
    ):

        print(
            f"[warn] Invalid Gemini response "
            f"for {qtype}",
            flush=True
        )

        return {
            "type": qtype,
            "error": "Invalid Gemini response",
            "raw": raw[:500]
        }

    # --------------------------------------------------------
    # MCQ
    # --------------------------------------------------------

    if qtype == "mcq":

        return {
            "type": "mcq",
            "question": parsed.get(
                "question",
                ""
            ),
            "choices": parsed.get(
                "choices",
                {}
            ),
            "answer_letter": parsed.get(
                "answer_letter",
                ""
            ),
            "answer_text": parsed.get(
                "answer_text",
                parsed["choices"][
                    parsed["answer_letter"]
                ]
            ),
            "explanation": parsed.get(
                "explanation",
                ""
            )
        }

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    if qtype == "short":

        return {
            "type": "short",
            "question": parsed.get(
                "question",
                ""
            ),
            "answer": parsed.get(
                "answer",
                ""
            ),
            "explanation": parsed.get(
                "explanation",
                ""
            )
        }

    # --------------------------------------------------------
    # FILL BLANK
    # --------------------------------------------------------

    if qtype == "fillblank":

        return {
            "type": "fillblank",
            "question": parsed.get(
                "question",
                ""
            ),
            "answer": parsed.get(
                "answer",
                ""
            ),
            "explanation": parsed.get(
                "explanation",
                ""
            )
        }

    # --------------------------------------------------------
    # TRUE / FALSE
    # --------------------------------------------------------

    if qtype == "tf":

        return {
            "type": "tf",
            "question": parsed.get(
                "question",
                ""
            ),
            "answer": parsed.get(
                "answer",
                ""
            ),
            "explanation": parsed.get(
                "explanation",
                ""
            )
        }

    raise ValueError(
        f"Unsupported question type: {qtype}"
    )


# ============================================================
# QUIZ GENERATION
# ============================================================

def generate_quiz(
    input_path: str,
    out_json: str = "quiz.json",
    max_questions: int = 50,
    question_types: List[str] = None,
    summarize_first: bool = True
):

    if question_types is None:

        question_types = [
            "mcq",
            "short",
            "fillblank",
            "tf"
        ]

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if max_questions < 1:

        raise ValueError(
            "max_questions must be at least 1"
        )

    if not question_types:

        raise ValueError(
            "At least one question type is required"
        )

    valid_types = {
        "mcq",
        "short",
        "fillblank",
        "tf"
    }

    question_types = [
        qt
        for qt in question_types
        if qt in valid_types
    ]

    if not question_types:

        raise ValueError(
            "No valid question types supplied"
        )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    print(
        "[start] Extracting text...",
        flush=True
    )

    raw_text = extract_text(
        input_path
    )

    text = clean_text(
        raw_text
    )

    if not text:

        raise ValueError(
            "No text could be extracted from input."
        )

    print(
        f"[info] Document length: "
        f"{len(text.split())} words",
        flush=True
    )

    # --------------------------------------------------------
    # Chunk
    # --------------------------------------------------------

    chunks = chunk_text(
        text
    )

    print(
        f"[info] Chunk count: {len(chunks)}",
        flush=True
    )

    if not chunks:

        raise ValueError(
            "No chunks were created."
        )

    # --------------------------------------------------------
    # Build TF-IDF retrieval index
    # --------------------------------------------------------

    vector_store = TfidfStore()

    vector_store.build(
        chunks
    )

    # --------------------------------------------------------
    # Create output structure
    # --------------------------------------------------------

    quiz = {
        "source": input_path,
        "questions": [],
        "summary": None
    }

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    if summarize_first:

        print(
            "[info] Generating summary...",
            flush=True
        )

        # Use first few chunks for summary
        summary_chunks = chunks[
            :min(4, len(chunks))
        ]

        summary_context = "\n\n".join(
            summary_chunks
        )

        try:

            quiz["summary"] = call_gemini(
                PROMPT_SUMMARY.replace(
                    "{context}", summary_context
                )
            )

        except Exception as e:

            print(
                f"[warn] Summary generation failed: {e}",
                flush=True
            )

            quiz["summary"] = None

    # --------------------------------------------------------
    # Question generation
    # --------------------------------------------------------

    qcount = 0

    chunk_idx = 0

    type_idx = 0

    # --------------------------------------------------------
    # We continue until we actually have the requested
    # number of valid questions.
    # --------------------------------------------------------

    while (
        qcount < max_questions
        and chunk_idx < len(chunks)
    ):

        current_chunk = chunks[
            chunk_idx
        ]

        current_type = question_types[
            type_idx % len(question_types)
        ]

        print(
            f"[info] Generating "
            f"{current_type} "
            f"using chunk {chunk_idx + 1}/"
            f"{len(chunks)}",
            flush=True
        )

        try:

            # ------------------------------------------------
            # RAG retrieval
            # ------------------------------------------------

            hits = vector_store.search(
                current_chunk,
                top_k=3
            )

            candidate_chunks = []

            for idx, score in hits:

                candidate_chunks.append(
                    vector_store.get_text(idx)
                )

            # ------------------------------------------------
            # Build context
            # ------------------------------------------------

            if candidate_chunks:

                context = "\n\n".join(
                    candidate_chunks
                )

            else:

                context = current_chunk

            # ------------------------------------------------
            # Generate question
            # ------------------------------------------------

            item = generate_one_question(
                context,
                current_type
            )

            # ------------------------------------------------
            # Only count valid questions
            # ------------------------------------------------

            if "error" not in item:

                item[
                    "source_chunk_index"
                ] = chunk_idx

                quiz[
                    "questions"
                ].append(item)

                qcount += 1

                print(
                    f"[ok] Generated "
                    f"{current_type} "
                    f"#{qcount}/{max_questions}",
                    flush=True
                )

            else:

                print(
                    f"[warn] Failed to generate "
                    f"valid {current_type}",
                    flush=True
                )

        except Exception as e:

            print(
                f"[warn] Generation failed "
                f"for chunk {chunk_idx}, "
                f"type {current_type}: {e}",
                flush=True
            )

        # ----------------------------------------------------
        # Move to next question type
        # ----------------------------------------------------

        type_idx += 1

        # ----------------------------------------------------
        # After using every selected type,
        # move to the next document chunk.
        # ----------------------------------------------------

        if (
            type_idx
            % len(question_types)
            == 0
        ):

            chunk_idx += 1

    # --------------------------------------------------------
    # Important: warn if document was exhausted
    # --------------------------------------------------------

    if qcount < max_questions:

        print(
            f"[warn] Document chunks exhausted. "
            f"Generated only {qcount}/"
            f"{max_questions} questions.",
            flush=True
        )

    # --------------------------------------------------------
    # Save quiz
    # --------------------------------------------------------

    with open(
        out_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            quiz,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"[done] Saved "
        f"{len(quiz['questions'])} questions "
        f"to {out_json}",
        flush=True
    )

    return quiz


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "RAG Quiz Generator "
            "(Gemini + TF-IDF)"
        )
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path or URL to input document"
    )

    parser.add_argument(
        "--out",
        "-o",
        default="quiz.json",
        help="Output JSON path"
    )

    parser.add_argument(
        "--max_questions",
        type=int,
        default=20,
        help="Number of questions to generate"
    )

    parser.add_argument(
        "--types",
        type=str,
        default="mcq,short,fillblank,tf",
        help=(
            "Comma-separated question types"
        )
    )

    parser.add_argument(
        "--no_summary",
        action="store_true",
        help="Disable summary generation"
    )

    args = parser.parse_args()

    question_types = [
        t.strip()
        for t in args.types.split(",")
        if t.strip()
    ]

    generate_quiz(
        input_path=args.input,
        out_json=args.out,
        max_questions=args.max_questions,
        question_types=question_types,
        summarize_first=(
            not args.no_summary
        )
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
