#!/usr/bin/env python3
"""
Optimized RAG Quiz Generator (Gemini-backed) - FIXED QUESTION COUNT
"""

import os
import re
import json
import time
import random
import argparse
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Optional

import requests
import pdfplumber
from docx import Document
from bs4 import BeautifulSoup

import nltk
import spacy

from sentence_transformers import SentenceTransformer, util, CrossEncoder
import faiss
import google.generativeai as genai

try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except Exception:
    KEYBERT_AVAILABLE = False

# ---------- Config ----------
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
MAX_GEN_RETRIES = 3
BACKOFF_BASE = 1.5
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Setup NLP
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

@lru_cache(maxsize=1)
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")
nlp = load_spacy()

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not set.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# ---------- Utilities ----------
def retry_with_backoff(fn, retries=MAX_GEN_RETRIES, base=BACKOFF_BASE):
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == retries:
                raise
            sleep_t = base ** attempt
            print(f"[retry] attempt {attempt} failed: {e}. sleeping {sleep_t:.1f}s")
            time.sleep(sleep_t)

# ---------- JSON Cleaning Function ----------
def clean_gemini_json(raw_text: str) -> Optional[dict]:
    """
    Parse Gemini output that may contain:
    - Markdown code fences (```json ... ```)
    - Plain JSON
    - Mixed text with JSON
    """
    if not raw_text:
        return None
    
    # Remove markdown code fences
    cleaned = re.sub(r'```json\s*', '', raw_text)
    cleaned = re.sub(r'```\s*', '', cleaned)
    cleaned = cleaned.strip()
    
    # Try to extract JSON if there's extra text
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(0)
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"[warn] JSON parse error: {e}")
        print(f"[warn] Raw text: {raw_text[:200]}...")
        return None

# ---------- Extraction ----------
def extract_text_from_pdf(path: str) -> str:
    parts = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            txt = p.extract_text()
            if txt:
                 parts.append(txt)
    return "\n\n".join(parts)

def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    return "\n\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip())

def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_text_from_url(url: str) -> str:
    def _get():
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.text
    html = retry_with_backoff(_get)
    soup = BeautifulSoup(html, "html.parser")
    paras = [p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()]
    return "\n\n".join(paras)

def extract_text(path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return extract_text_from_url(path_or_url)
    p = Path(path_or_url)
    if not p.exists():
        raise FileNotFoundError(f"Input {path_or_url} not found.")
    ext = p.suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path_or_url)
    elif ext in [".docx", ".doc"]:
        return extract_text_from_docx(path_or_url)
    elif ext == ".txt":
        return extract_text_from_txt(path_or_url)
    else:
        raise ValueError("Unsupported input format.")

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"(?im)^(page|pg)\s*\d+\b.*$", "", text)
    text = re.sub(r"(?im)^(confidential|copyright).*$", "", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\[[0-9]+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------- Chunking ----------
def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_words]
        chunks.append(" ".join(chunk))
        i += chunk_words - overlap
    return chunks

# ---------- FAISS ----------
class FaissStore:
    def __init__(self, embed_model_name: str = EMBED_MODEL):
        print(f"[info] Loading embedding model: {embed_model_name}", flush=True)
        self.model = SentenceTransformer(embed_model_name)
        self.index = None
        self.chunks: List[str] = []

    def build(self, chunks: List[str]):
        if not chunks:
            self.index = None
            self.chunks = []
            return
        print(f"[info] Encoding {len(chunks)} chunks...", flush=True)
        embs = self.model.encode(chunks, show_progress_bar=True, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(embs)
        d = embs.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embs)
        self.chunks = chunks
        print(f"[info] FAISS index built (dim={d})", flush=True)

    def search(self, query: str, top_k: int = 8) -> List[Tuple[int, float]]:
        if not self.index:
            return []
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((int(idx), float(score)))
        return results

    def get_text(self, idx: int) -> str:
        return self.chunks[idx]

# ---------- Reranker ----------
class Reranker:
    def __init__(self, model_name: str = CROSS_ENCODER_MODEL, top_n: int = 3):
        print(f"[info] Loading cross-encoder: {model_name}", flush=True)
        self.model = CrossEncoder(model_name)
        self.top_n = top_n

    def rerank(self, query: str, candidate_texts: List[str]) -> List[str]:
        if not candidate_texts:
            return []
        pairs = [[query, t] for t in candidate_texts]
        scores = self.model.predict(pairs)
        scored = sorted(zip(candidate_texts, scores), key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored[: self.top_n]]

# ---------- Candidate extraction ----------
def extract_candidates(text: str, top_n: int = 20) -> List[str]:
    candidates = []
    if KEYBERT_AVAILABLE:
        try:
            kw = KeyBERT(model=EMBED_MODEL)
            kws = kw.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words="english", top_n=top_n)
            candidates.extend([k[0] for k in kws])
        except Exception:
            pass
    doc = nlp(text)
    for nc in doc.noun_chunks:
        s = nc.text.strip()
        if 1 < len(s.split()) <= 6:
            candidates.append(s)
    for ent in doc.ents:
        s = ent.text.strip()
        if 1 < len(s.split()) <= 6:
            candidates.append(s)
    seen = set()
    uniq = []
    for c in candidates:
        lc = c.lower()
        if lc not in seen:
            seen.add(lc)
            uniq.append(c)
    return uniq[:top_n]

# ---------- Gemini call ----------
def call_gemini(prompt: str, model_name: str = GEMINI_MODEL, temperature: float = 0.0) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    model_name = model_name.replace("models/", "").strip()
    def _call():
        mdl = genai.GenerativeModel(model_name)
        resp = mdl.generate_content(prompt)
        return resp.text if getattr(resp, "text", None) else ""
    return retry_with_backoff(_call)

# ---------- Prompts ----------
PROMPT_SUMMARY = (
    "Produce 3-5 short bullet points (10-25 words each) summarizing:\n\n{context}\n\nOUTPUT:"
)

PROMPT_MCQ = (
    "Create ONE multiple-choice question with 4 options (A, B, C, D).\n"
    "Return ONLY valid JSON with these exact keys:\n"
    '- "question": the question text\n'
    '- "choices": object with keys A, B, C, D\n'
    '- "answer_letter": correct letter (A/B/C/D)\n'
    '- "answer_text": text of correct answer\n'
    '- "explanation": one sentence why it\'s correct\n\n'
    "CONTEXT:\n{context}\n\nJSON:"
)

PROMPT_SHORT = (
    "Create ONE short-answer question.\n"
    "Return ONLY valid JSON with these exact keys:\n"
    '- "question": the question text\n'
    '- "answer": the correct answer\n'
    '- "explanation": one sentence explanation\n\n'
    "CONTEXT:\n{context}\n\nJSON:"
)

PROMPT_FILL = (
    "Create ONE fill-in-the-blank question with '_____'.\n"
    "Return ONLY valid JSON with these exact keys:\n"
    '- "question": the question with blank\n'
    '- "answer": word/phrase for blank\n'
    '- "explanation": one sentence explanation\n\n'
    "CONTEXT:\n{context}\n\nJSON:"
)

PROMPT_TF = (
    "Create ONE True/False question.\n"
    "Return ONLY valid JSON with these exact keys:\n"
    '- "question": the statement\n'
    '- "answer": "True" or "False"\n'
    '- "explanation": one sentence explanation\n\n'
    "CONTEXT:\n{context}\n\nJSON:"
)

# ---------- Question generation ----------
def generate_one_question(context: str, qtype: str, emb_model: SentenceTransformer, reranker_ce: Optional[CrossEncoder]=None) -> dict:
    ctx = context.strip()
    if not ctx:
        return {"type": qtype, "error": "empty context"}

    if qtype == "summary":
        raw = call_gemini(PROMPT_SUMMARY.format(context=ctx))
        return {"type": "summary", "text": raw}

    # For all question types, use JSON prompts
    if qtype == "mcq":
        raw = call_gemini(PROMPT_MCQ.format(context=ctx))
        parsed = clean_gemini_json(raw)
        if parsed and all(k in parsed for k in ["question", "choices", "answer_letter"]):
            return {
                "type": "mcq",
                "question": parsed.get("question", ""),
                "choices": parsed.get("choices", {}),
                "answer_letter": parsed.get("answer_letter", ""),
                "answer_text": parsed.get("answer_text", ""),
                "explanation": parsed.get("explanation", "")
            }
        # Fallback if parsing fails
        return {"type": "mcq", "question": "Parse error", "choices": {}, "answer_letter": "A", "answer_text": "", "explanation": raw[:200]}

    elif qtype == "short":
        raw = call_gemini(PROMPT_SHORT.format(context=ctx))
        parsed = clean_gemini_json(raw)
        if parsed and "question" in parsed:
            return {
                "type": "short",
                "question": parsed.get("question", ""),
                "answer": parsed.get("answer", ""),
                "explanation": parsed.get("explanation", "")
            }
        return {"type": "short", "question": "Parse error", "answer": "", "explanation": raw[:200]}

    elif qtype == "fillblank":
        raw = call_gemini(PROMPT_FILL.format(context=ctx))
        parsed = clean_gemini_json(raw)
        if parsed and "question" in parsed:
            return {
                "type": "fillblank",
                "question": parsed.get("question", ""),
                "answer": parsed.get("answer", ""),
                "explanation": parsed.get("explanation", "")
            }
        return {"type": "fillblank", "question": "Parse error", "answer": "", "explanation": raw[:200]}

    elif qtype == "tf":
        raw = call_gemini(PROMPT_TF.format(context=ctx))
        parsed = clean_gemini_json(raw)
        if parsed and "question" in parsed:
            return {
                "type": "tf",
                "question": parsed.get("question", ""),
                "answer": parsed.get("answer", ""),
                "explanation": parsed.get("explanation", "")
            }
        return {"type": "tf", "question": "Parse error", "answer": "", "explanation": raw[:200]}

    raise ValueError("Unsupported qtype")

# ---------- Orchestrator ----------
def generate_quiz(input_path: str, out_json: str = "quiz.json", max_questions: int = 50,
                  question_types: List[str] = ["mcq", "short", "fillblank", "tf"], summarize_first: bool = True):
    print("[start] Extracting text...", flush=True)
    raw = extract_text(input_path)
    text = clean_text(raw)
    if not text:
        raise ValueError("No text extracted.")
    print(f"[info] Document length: {len(text.split())} words", flush=True)

    chunks = chunk_text(text)
    print(f"[info] Chunk count: {len(chunks)}", flush=True)

    vs = FaissStore(EMBED_MODEL)
    vs.build(chunks)

    reranker = Reranker(CROSS_ENCODER_MODEL, top_n=3)
    embed_model = vs.model

    quiz = {"source": input_path, "questions": [], "summary": None}
    if summarize_first and chunks:
        seed_ctx = " ".join(chunks[:min(4, len(chunks))])
        quiz["summary"] = call_gemini(PROMPT_SUMMARY.format(context=seed_ctx))

    # FIXED: Generate questions in a round-robin fashion until we reach max_questions
    qcount = 0
    chunk_idx = 0
    type_idx = 0
    
    while qcount < max_questions and chunk_idx < len(chunks):
        chunk = chunks[chunk_idx]
        qtype = question_types[type_idx % len(question_types)]
        
        try:
            hits = vs.search(chunk, top_k=8)
            cand_texts = [vs.get_text(idx) for idx, _ in hits]
            top_ctxs = reranker.rerank(chunk, cand_texts) if cand_texts else [chunk]
            combined_ctx = "\n\n".join(top_ctxs)
            
            item = generate_one_question(combined_ctx, qtype, embed_model, reranker.model)
            item["source_chunk_index"] = chunk_idx
            quiz["questions"].append(item)
            qcount += 1
            print(f"[ok] Generated {qtype} #{qcount}/{max_questions}", flush=True)
        except Exception as e:
            print(f"[warn] generation failed for chunk {chunk_idx}, qtype {qtype}: {e}", flush=True)
        
        # Move to next question type
        type_idx += 1
        
        # If we've gone through all question types, move to next chunk
        if type_idx % len(question_types) == 0:
            chunk_idx += 1

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(quiz, f, ensure_ascii=False, indent=2)
    print(f"[done] Saved {len(quiz['questions'])} questions to {out_json}", flush=True)

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Optimized RAG Quiz Generator (Gemini)")
    parser.add_argument("--input", "-i", required=True, help="Path or URL to input doc")
    parser.add_argument("--out", "-o", default="quiz.json", help="Output JSON file")
    parser.add_argument("--max_questions", type=int, default=20)
    parser.add_argument("--types", type=str, default="mcq,short,fillblank,tf")
    parser.add_argument("--no_summary", action="store_true", help="Disable summary")
    args = parser.parse_args()

    qtypes = [t.strip() for t in args.types.split(",") if t.strip()]
    generate_quiz(args.input, args.out, args.max_questions, qtypes, summarize_first=not args.no_summary)

if __name__ == "__main__":
    main()