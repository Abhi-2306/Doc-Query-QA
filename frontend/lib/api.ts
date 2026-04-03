/**
 * api.ts — All FastAPI calls in one place.
 * Components import functions from here, never call fetch directly.
 */

const BASE_URL = "/api"; // proxied to http://localhost:8000 via next.config.ts

// ── Types ────────────────────────────────────────────────

export interface ModelResult {
  answer:      string;
  confidence:  number;
  latency_ms:  number;
}

export interface AskResponse {
  question:   string;
  bert:       ModelResult;
  distilbert: ModelResult;
  best_chunk: string;
}

export interface UploadResponse {
  message:       string;
  num_chunks:    number;
  num_questions: number;
}

export interface AutoQAResponse {
  questions: string[];
}

export interface HealthResponse {
  status:         string;
  distilbert:     boolean;
  bert:           boolean;
  chunks_loaded:  number;
}

// ── API functions ────────────────────────────────────────

/**
 * Upload a PDF file to the backend.
 * Triggers text extraction, chunking, and question generation.
 */
export async function uploadPDF(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BASE_URL}/upload`, {
    method: "POST",
    body:   formData,
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Failed to upload PDF");
  }

  return res.json();
}

/**
 * Ask a question about the uploaded document.
 * Returns answers from both BERT and DistilBERT.
 */
export async function askQuestion(question: string): Promise<AskResponse> {
  const res = await fetch(`${BASE_URL}/ask`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ question }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Failed to get answer");
  }

  return res.json();
}

/**
 * Get the auto-generated suggested questions for the uploaded document.
 */
export async function getAutoQA(): Promise<AutoQAResponse> {
  const res = await fetch(`${BASE_URL}/auto-qa`);

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Failed to fetch questions");
  }

  return res.json();
}

/**
 * Health check — verify which models are loaded.
 */
export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE_URL}/health`);
  if (!res.ok) throw new Error("Backend not reachable");
  return res.json();
}