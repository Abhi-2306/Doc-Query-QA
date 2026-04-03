"""
Evaluate DistilBERT Final Model
Measures Exact Match, F1, and Speed on 1000 SQuAD validation examples.
Run this after stage2_coqa.py is complete.

Usage (Colab):
    python evaluate_distilbert.py

Usage (local — after dropping distilbert-final/ into models/):
    py evaluate_distilbert.py
"""

import time
import json
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, set_seed
from datasets import load_dataset
from evaluate import load as load_metric
import torch

set_seed(42)

# ── Config ──────────────────────────────────────────────
MODEL_DIR   = "./distilbert-final"   # change to ../models/distilbert-final when running locally
EVAL_SIZE   = 1000                   # same slice Sruthi uses — keeps comparison fair
MAX_LENGTH  = 384
DOC_STRIDE  = 128
# ────────────────────────────────────────────────────────


def get_answer(question: str, context: str, tokenizer, model) -> tuple[str, float]:
    """Run model on a single question+context, return (answer, latency_ms)."""
    inputs = tokenizer(
        question,
        context,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    latency = (time.time() - start) * 1000  # ms

    start_idx = torch.argmax(outputs.start_logits)
    end_idx   = torch.argmax(outputs.end_logits) + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0][start_idx:end_idx]
        )
    ).strip()

    return answer or "no answer", latency


def main():
    print("=== Evaluate DistilBERT Final Model ===\n")

    # 1. Load model
    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model     = DistilBertForQuestionAnswering.from_pretrained(MODEL_DIR)
    model.eval()

    # 2. Load eval data — same 1000 examples Sruthi uses
    print("Loading SQuAD validation slice (1000 examples)...")
    dataset = load_dataset("squad")
    eval_data = dataset["validation"].select(range(EVAL_SIZE))

    # 3. Run evaluation
    metric     = load_metric("squad")
    predictions = []
    references  = []
    latencies   = []

    print(f"Running inference on {EVAL_SIZE} examples...")
    for i, ex in enumerate(eval_data):
        if i % 100 == 0:
            print(f"  {i}/{EVAL_SIZE}...")

        answer, latency = get_answer(ex["question"], ex["context"], tokenizer, model)
        latencies.append(latency)

        predictions.append({
            "id":         ex["id"],
            "prediction_text": answer,
        })
        references.append({
            "id":      ex["id"],
            "answers": ex["answers"],
        })

    # 4. Compute metrics
    results = metric.compute(predictions=predictions, references=references)
    avg_latency = sum(latencies) / len(latencies)

    # 5. Print results
    print("\n" + "="*45)
    print("         DISTILBERT EVALUATION RESULTS")
    print("="*45)
    print(f"  Exact Match : {results['exact_match']:.2f}%")
    print(f"  F1 Score    : {results['f1']:.2f}%")
    print(f"  Avg Latency : {avg_latency:.2f} ms/query")
    print(f"  Model Size  : 66M parameters")
    print("="*45)

    # 6. Save results to JSON for the comparison report
    output = {
        "model":        "DistilBERT",
        "eval_size":    EVAL_SIZE,
        "exact_match":  round(results["exact_match"], 2),
        "f1":           round(results["f1"], 2),
        "avg_latency_ms": round(avg_latency, 2),
        "parameters":   "66M",
    }

    with open("distilbert_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n✅ Results saved to distilbert_results.json")


if __name__ == "__main__":
    main()