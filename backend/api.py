"""
FastAPI Backend — DocQuery
Connects Sruthi's pipeline with Abhijith's models and the Next.js frontend.

Endpoints:
  POST /upload   — accepts PDF, extracts + chunks text, generates suggested questions
  POST /ask      — takes question, runs both BERT + DistilBERT, returns both answers
  GET  /auto-qa  — returns the suggested questions generated on upload
"""

import sys
import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForQuestionAnswering,
    BertTokenizerFast,
    BertForQuestionAnswering,
)
import torch
import time

# ── Pipeline imports (Sruthi's modules) ─────────────────
# TODO: Uncomment these once Sruthi's files are ready
# sys.path.append(str(Path(__file__).parent.parent / "pipeline"))
# from pdf_processor import extract_text, chunk_text
# from retriever import find_best_chunk
# from conversation import build_context, save_to_history, clear_history

# Placeholder functions — remove once Sruthi's modules are integrated
def extract_text(pdf_bytes: bytes) -> str:
    return "Placeholder extracted text from PDF."

def chunk_text(text: str) -> list[str]:
    # Simple word-based chunking as placeholder
    words  = text.split()
    size   = 400
    stride = 50
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - stride
    return chunks if chunks else [text]

def find_best_chunk(question: str, chunks: list[str]) -> str:
    return chunks[0] if chunks else ""

def build_context(question: str, chunk: str) -> str:
    return chunk

def save_to_history(question: str, bert_ans: str, distil_ans: str):
    pass

def clear_history():
    pass
# ────────────────────────────────────────────────────────

from auto_qa import generate_suggested_questions

# ── Config ──────────────────────────────────────────────
BASE           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISTILBERT_DIR = os.path.join(BASE, "models", "distilbert-final")
BERT_DIR       = os.path.join(BASE, "models", "bert-final")
# ────────────────────────────────────────────────────────

app = FastAPI(title="DocQuery API")

# Allow Next.js frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory state ──────────────────────────────────────
state = {
    "chunks":     [],   # text chunks from uploaded PDF
    "auto_qa":    [],   # suggested questions
}
# ────────────────────────────────────────────────────────


# ── Load models on startup ───────────────────────────────
print("Loading DistilBERT model...")
try:
    distil_tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_DIR)
    distil_model     = DistilBertForQuestionAnswering.from_pretrained(DISTILBERT_DIR)
    distil_model.eval()
    print("✅ DistilBERT loaded")
except Exception as e:
    print(f"⚠️  DistilBERT not found at {DISTILBERT_DIR} — train Stage 1 + 2 first. ({e})")
    distil_tokenizer = None
    distil_model     = None

print("Loading BERT model...")
try:
    bert_tokenizer = BertTokenizerFast.from_pretrained(BERT_DIR)
    bert_model     = BertForQuestionAnswering.from_pretrained(BERT_DIR)
    bert_model.eval()
    print("✅ BERT loaded")
except Exception as e:
    print(f"⚠️  BERT not found at {BERT_DIR} — Sruthi's training must complete first. ({e})")
    bert_tokenizer = None
    bert_model     = None
# ────────────────────────────────────────────────────────


def run_qa(question: str, context: str, tokenizer, model) -> dict:
    """Run a single QA model and return answer + confidence + latency."""
    if model is None or tokenizer is None:
        return {"answer": "Model not loaded", "confidence": 0.0, "latency_ms": 0}

    start = time.time()

    inputs = tokenizer(
        question,
        context,
        max_length=384,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = torch.argmax(outputs.start_logits)
    end_idx   = torch.argmax(outputs.end_logits) + 1

    # Confidence: average softmax of start + end logits
    start_conf = torch.softmax(outputs.start_logits, dim=1).max().item()
    end_conf   = torch.softmax(outputs.end_logits,   dim=1).max().item()
    confidence = round((start_conf + end_conf) / 2, 4)

    answer  = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx])
    ).strip()
    latency = round((time.time() - start) * 1000, 2)  # ms

    return {"answer": answer or "No answer found", "confidence": confidence, "latency_ms": latency}


# ── Request / Response schemas ───────────────────────────
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question:    str
    bert:        dict
    distilbert:  dict
    best_chunk:  str


# ── Endpoints ────────────────────────────────────────────
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Accepts PDF → extracts text → chunks → generates suggested questions."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()

    # Extract + chunk
    text   = extract_text(pdf_bytes)
    chunks = chunk_text(text)

    # Clear previous session
    clear_history()
    state["chunks"]  = chunks
    state["auto_qa"] = []   # reset — will generate fresh

    # Generate suggested questions in background
    state["auto_qa"] = generate_suggested_questions(chunks)

    return {
        "message":    "PDF uploaded successfully",
        "num_chunks": len(chunks),
        "num_questions": len(state["auto_qa"]),
    }


@app.post("/ask", response_model=AskResponse)
async def ask_question(body: AskRequest):
    """Takes a question → retrieves best chunk → runs both models → returns both answers."""
    if not state["chunks"]:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet.")

    question   = body.question.strip()
    best_chunk = find_best_chunk(question, state["chunks"])
    context    = build_context(question, best_chunk)

    bert_result   = run_qa(question, context, bert_tokenizer,   bert_model)
    distil_result = run_qa(question, context, distil_tokenizer, distil_model)

    save_to_history(question, bert_result["answer"], distil_result["answer"])

    return AskResponse(
        question   = question,
        bert       = bert_result,
        distilbert = distil_result,
        best_chunk = best_chunk,
    )


@app.get("/auto-qa")
async def get_auto_qa():
    """Returns the suggested questions generated from the uploaded PDF."""
    if not state["auto_qa"]:
        raise HTTPException(status_code=404, detail="No questions yet — upload a PDF first.")
    return {"questions": state["auto_qa"]}


@app.get("/health")
async def health():
    return {
        "status":      "ok",
        "distilbert":  distil_model is not None,
        "bert":        bert_model is not None,
        "chunks_loaded": len(state["chunks"]),
    }