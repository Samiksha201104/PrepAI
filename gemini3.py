#!/usr/bin/env python3
"""
Optimized RAG Quiz Generator (Gemini-backed)
- Retains all features: PDF/DOCX/TXT/URL ingestion, cleaning, chunking,
  embeddings (SBERT), FAISS, cross-encoder reranker, KeyBERT+spaCy candidates,
  distractor selection, and Gemini generation.
- Improvements: retries, backoff, deterministic behavior, robust parsing,
  cross-encoder ranking for distractors, better prompts, light caching.

Usage:
    export GEMINI_API_KEY="..."
    python rag_quiz_generator_gemini_opt.py --input sample.pdf --out quiz.json --max_questions 20
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

# embeddings & cross-encoder
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import faiss

# Gemini
import google.generativeai as genai

# Optional KeyBERT
try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except Exception:
    KEYBERT_AVAILABLE = False

# ---------- Config / Tunables ----------
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# prefer free-ish gemini model; allow override via env
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
MAX_GEN_RETRIES = 3
BACKOFF_BASE = 1.5
RANDOM_SEED = 42  # deterministic shuffle for reproducibility
random.seed(RANDOM_SEED)
# ---------------------------------------

# setup minimal NLP downloads
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

# Configure Gemini
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not set. Set GEMINI_API_KEY env var before running for Gemini generation.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# ---------- Utilities: resilient HTTP / retries ----------
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

# ---------- Extraction & cleaning ----------
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
        raise ValueError("Unsupported input format. Use pdf/docx/txt or http(s) url.")

def clean_text(text: str) -> str:
    if not text:
        return ""
    # preserve paragraphs but normalize whitespace
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"(?im)^(page|pg)\s*\d+\b.*$", "", text)
    text = re.sub(r"(?im)^(confidential|copyright).*$", "", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\[[0-9]+\]", "", text)
    # collapse multiple spaces & trim
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

# ---------- FAISS + SBERT ----------
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

# ---------- CrossEncoder Reranker ----------
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

# ---------- Candidate extraction (KeyBERT + spaCy) ----------
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
    # dedupe preserving order
    seen = set()
    uniq = []
    for c in candidates:
        lc = c.lower()
        if lc not in seen:
            seen.add(lc)
            uniq.append(c)
    return uniq[:top_n]

# ---------- Distractor selection: cross-encoder + SBERT fallback ----------
def rank_distractors(answer: str, candidates: List[str], embed_model: SentenceTransformer, cross_encoder: Optional[CrossEncoder]=None, top_k: int = 3) -> List[str]:
    if not candidates or not answer:
        return []
    # remove identical
    filtered = [c for c in candidates if c.lower() != answer.lower()]
    if not filtered:
        return []
    # Prefer cross-encoder scoring if available (better semantic ranking)
    if cross_encoder is not None:
        pairs = [[answer, c] for c in filtered]
        try:
            scores = cross_encoder.predict(pairs)
            scored = sorted(zip(filtered, scores), key=lambda x: x[1], reverse=True)
            # pick top candidates but avoid near-identical by SBERT cosine
            sbert_embs = embed_model.encode([answer] + [c for c, _ in scored], convert_to_tensor=True)
            ans_emb = sbert_embs[0]
            cand_embs = sbert_embs[1:]
            sims = util.cos_sim(ans_emb, cand_embs)[0]
            chosen = []
            for i, (c, sc) in enumerate(scored):
                if len(chosen) >= top_k:
                    break
                if sims[i] > 0.995:
                    continue
                chosen.append(c)
            return chosen
        except Exception:
            pass
    # fallback SBERT similarity: choose close-but-not-identical
    texts = [answer] + filtered
    embs = embed_model.encode(texts, convert_to_tensor=True)
    ans_emb = embs[0]
    cand_embs = embs[1:]
    sims = util.cos_sim(ans_emb, cand_embs)[0]
    pairs = [(filtered[i], float(sims[i])) for i in range(len(filtered))]
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
    distractors = []
    for cand, sim in pairs_sorted:
        if len(distractors) >= top_k:
            break
        if sim > 0.995:  # skip near-identical
            continue
        distractors.append(cand)
    return distractors

# ---------- Robust parsing helpers ----------
MCQ_CHOICE_RE = re.compile(r"^([A-D])[\)\.\-]\s*(.+)$", re.IGNORECASE)
ANSWER_RE = re.compile(r"^(?:answer|correct)\s*[:\-]\s*([A-D]|[A-D]\))", re.IGNORECASE)
EXPLANATION_RE = re.compile(r"^explanation\s*[:\-]\s*(.+)$", re.IGNORECASE)

def parse_mcq_text(raw: str) -> Optional[dict]:
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    q = None
    choices = {}
    answer_letter = None
    explanation = None
    for l in lines:
        if l.lower().startswith("q:") or l.lower().startswith("question:"):
            q = l.split(":", 1)[1].strip()
            continue
        m_choice = MCQ_CHOICE_RE.match(l)
        if m_choice:
            choices[m_choice.group(1).upper()] = m_choice.group(2).strip()
            continue
        m_ans = ANSWER_RE.match(l)
        if m_ans:
            answer_letter = m_ans.group(1).upper().strip().replace(")", "")
            continue
        m_expl = EXPLANATION_RE.match(l)
        if m_expl:
            explanation = m_expl.group(1).strip()
    # fallback: sometimes answer line is like "Answer: B) text"
    if not answer_letter:
        for l in lines:
            if l.lower().startswith("answer:"):
                rest = l.split(":", 1)[1].strip()
                if rest:
                    candidate = rest.split()[0].strip().upper().replace(")", "").replace(".", "")
                    if candidate in choices:
                        answer_letter = candidate
                        break
    if q and choices and answer_letter:
        return {
            "type": "mcq",
            "question": q,
            "choices": choices,
            "answer_letter": answer_letter,
            "answer_text": choices.get(answer_letter),
            "explanation": explanation
        }
    return None

# ---------- Gemini generation wrapper (safe) ----------
def call_gemini(prompt: str, model_name: str = GEMINI_MODEL, temperature: float = 0.0) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set in env")
    # ensure model name doesn't have 'models/' prefix
    model_name = model_name.replace("models/", "").strip()
    def _call():
        mdl = genai.GenerativeModel(model_name)
        # generate_content should accept a prompt string; keep generation concise
        resp = mdl.generate_content(prompt)
        return resp.text if getattr(resp, "text", None) else ""
    return retry_with_backoff(_call)

# ---------- Question generation ----------

# improved, consistent prompts with clear format expectations
PROMPT_SUMMARY = (
    "You are a concise summarizer. Produce 3-5 short bullet points (each 10-25 words) summarizing the facts in the CONTEXT.\n\n"
    "CONTEXT:\n{context}\n\nBULLETS:\n"
)
PROMPT_MCQ = (
    "You are a careful quiz-maker. Using the CONTEXT produce ONE multiple-choice question with four options labeled A) B) C) D).\n"
    "Mark the correct option with 'Answer: <letter>' and include 'Explanation: <one-line explanation>'.\n\nCONTEXT:\n{context}\n\nOUTPUT:\n"
)
PROMPT_SHORT = (
    "Using the CONTEXT produce one short-answer question and a one-line ideal answer. Format:\nQ: <question>\nAnswer: <short answer>\nExplanation: <one-line>\n\nCONTEXT:\n{context}\n"
)
PROMPT_FILL = (
    "Using the CONTEXT produce one fill-in-the-blank (single blank '_____'), give the answer and one-line explanation.\n\nCONTEXT:\n{context}\n"
)
PROMPT_TF = (
    "Using the CONTEXT produce one True/False question. Provide statement, 'Answer: True' or 'Answer: False', and one-line explanation.\n\nCONTEXT:\n{context}\n"
)

def generate_one_question(context: str, qtype: str, emb_model: SentenceTransformer, reranker_ce: Optional[CrossEncoder]=None) -> dict:
    ctx = context.strip()
    if not ctx:
        return {"type": qtype, "error": "empty context"}

    if qtype == "summary":
        raw = call_gemini(PROMPT_SUMMARY.format(context=ctx))
        return {"type": "summary", "text": raw}

    if qtype == "mcq":
        raw = call_gemini(PROMPT_MCQ.format(context=ctx))
        parsed = parse_mcq_text(raw)
        if parsed:
            return parsed
        # fallback: ask gemini to output "Q: ... Answer: <answer>"
        qa_raw = call_gemini(f"From the CONTEXT below produce a single question and answer in format 'Q: ...\\nAnswer: ...'\\n\nCONTEXT:\\n{ctx}")
        q, a = None, None
        for line in qa_raw.splitlines():
            if line.lower().startswith("q:"):
                q = line.split(":",1)[1].strip()
            if line.lower().startswith("answer:"):
                a = line.split(":",1)[1].strip()
        if not q or not a:
            return {"type": "mcq", "raw": raw}
        # get candidate pool + pick distractors using cross-encoder prioritization
        cands = extract_candidates(ctx, top_n=30)
        distractors = rank_distractors(a, cands, emb_model, cross_encoder=reranker_ce, top_k=3)
        if len(distractors) < 3:
            # ask gemini for distractors
            extra = call_gemini(f"Provide 3 plausible distractors for the correct answer: '{a}' — comma separated.")
            extras = [p.strip() for p in re.split(r",|;|\\n", extra) if p.strip() and p.strip().lower() != a.lower()]
            for e in extras:
                if e not in distractors:
                    distractors.append(e)
                if len(distractors) >= 3:
                    break
        distractors = distractors[:3]
        all_choices = [a] + distractors
        random.Random(RANDOM_SEED).shuffle(all_choices)  # deterministic shuffle
        letters = ["A", "B", "C", "D"]
        choices = {letters[i]: all_choices[i] for i in range(len(all_choices))}
        ans_letter = letters[all_choices.index(a)]
        explanation = call_gemini(f"Explain in one sentence why the answer '{a}' is correct based on:\n{ctx}")
        return {"type": "mcq", "question": q, "choices": choices, "answer_letter": ans_letter, "answer_text": a, "explanation": explanation}

    if qtype == "short":
        raw = call_gemini(PROMPT_SHORT.format(context=ctx))
        q, a = None, None
        for line in raw.splitlines():
            if line.lower().startswith("q:"):
                q = line.split(":",1)[1].strip()
            elif line.lower().startswith("answer:"):
                a = line.split(":",1)[1].strip()
        # fallback: whole raw as explanation if no parse
        return {"type": "short", "question": q or raw.splitlines()[0] if raw else None, "answer": a, "explanation": raw}

    if qtype == "fillblank":
        raw = call_gemini(PROMPT_FILL.format(context=ctx))
        q, a = None, None
        for line in raw.splitlines():
            if line.lower().startswith("q:"):
                q = line.split(":",1)[1].strip()
            elif line.lower().startswith("answer:"):
                a = line.split(":",1)[1].strip()
        return {"type": "fillblank", "question": q or raw, "answer": a, "explanation": raw}

    if qtype == "tf":
        raw = call_gemini(PROMPT_TF.format(context=ctx))
        q, a = None, None
        for line in raw.splitlines():
            if line.lower().startswith("q:"):
                q = line.split(":",1)[1].strip()
            elif line.lower().startswith("answer:"):
                a = line.split(":",1)[1].strip()
        return {"type": "tf", "question": q or raw, "answer": a, "explanation": raw}

    raise ValueError("Unsupported qtype")

# ---------- Orchestrator ----------
def generate_quiz(input_path: str, out_json: str = "quiz.json", max_questions: int = 20,
                  question_types: List[str] = ["mcq", "short", "fillblank", "tf"], summarize_first: bool = True):
    print("[start] Extracting text...", flush=True)
    raw = extract_text(input_path)
    text = clean_text(raw)
    if not text:
        raise ValueError("No text extracted from input.")
    print(f"[info] Document length: {len(text.split())} words", flush=True)

    chunks = chunk_text(text)
    print(f"[info] Chunk count: {len(chunks)}", flush=True)

    # build vector store (SBERT embeddings)
    vs = FaissStore(EMBED_MODEL)
    vs.build(chunks)

    # load reranker & reuse for distractor ranking
    reranker = Reranker(CROSS_ENCODER_MODEL, top_n=3)
    embed_model = vs.model

    quiz = {"source": input_path, "questions": [], "summary": None}
    if summarize_first and chunks:
        seed_ctx = " ".join(chunks[:min(4, len(chunks))])
        quiz["summary"] = call_gemini(PROMPT_SUMMARY.format(context=seed_ctx))

    qcount = 0
    for i, chunk in enumerate(chunks):
        if qcount >= max_questions:
            break
        # retrieve top neighbors and rerank
        hits = vs.search(chunk, top_k=8)
        cand_texts = [vs.get_text(idx) for idx, _ in hits]
        top_ctxs = reranker.rerank(chunk, cand_texts) if cand_texts else [chunk]
        combined_ctx = "\n\n".join(top_ctxs)
        for qtype in question_types:
            if qcount >= max_questions:
                break
            try:
                item = generate_one_question(combined_ctx, qtype, embed_model, reranker.model)
                item["source_chunk_index"] = i
                item["source_excerpt"] = combined_ctx[:800]
                quiz["questions"].append(item)
                qcount += 1
                print(f"[ok] Generated {qtype} #{qcount}", flush=True)
            except Exception as e:
                print(f"[warn] generation failed for chunk {i}, qtype {qtype}: {e}", flush=True)
                continue

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
