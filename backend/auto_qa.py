"""
Auto Q&A Generation
Takes the first 5 chunks from an uploaded document and generates
3 suggested questions per chunk using a fine-tuned T5 model.
Output: list of question strings shown on the UI as clickable chips.
"""

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# ── Config ──────────────────────────────────────────────
MODEL_NAME = "mrm8488/t5-base-finetuned-question-generation-ap"
MAX_CHUNKS = 5  # only use first 5 chunks
QUESTIONS_PER_CHUNK = 3
MAX_INPUT_LENGTH = 512
MAX_QUESTION_LENGTH = 64
# ────────────────────────────────────────────────────────

# Load model once at module level so API doesn't reload on every request
print("Loading question generation model...")
_tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
_model.eval()
print("✅ Question generation model loaded")


def _generate_questions_from_chunk(chunk: str) -> list[str]:
    """Generate QUESTIONS_PER_CHUNK questions from a single text chunk."""
    # Model expects input in format: "answer: <text> context: <text>"
    # Since we want open-ended questions, we use the chunk as both
    input_text = f"answer: {chunk[:300]} context: {chunk}"

    inputs = _tokenizer(
        input_text,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = _model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=MAX_QUESTION_LENGTH,
            num_beams=QUESTIONS_PER_CHUNK + 1,
            num_return_sequences=QUESTIONS_PER_CHUNK,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )

    questions = [
        _tokenizer.decode(out, skip_special_tokens=True).strip()
        for out in outputs
    ]

    # Filter out empty or very short outputs
    questions = [
    q.replace("question:", "").strip()
    for q in questions if len(q) > 10
]
    return questions


def generate_suggested_questions(chunks: list[str]) -> list[str]:
    """
    Main function called by api.py after PDF upload.
    Takes list of text chunks, returns flat list of suggested questions.

    Args:
        chunks: list of text chunks from pdf_processor.py

    Returns:
        list of question strings (up to MAX_CHUNKS * QUESTIONS_PER_CHUNK)
    """
    if not chunks:
        return []

    selected_chunks = chunks[:MAX_CHUNKS]
    all_questions = []

    for i, chunk in enumerate(selected_chunks):
        print(f"Generating questions for chunk {i + 1}/{len(selected_chunks)}...")
        questions = _generate_questions_from_chunk(chunk)
        all_questions.extend(questions)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for q in all_questions:
        if q.lower() not in seen:
            seen.add(q.lower())
            unique.append(q)

    print(f"✅ Generated {len(unique)} suggested questions")
    return unique


# ── Quick test ──────────────────────────────────────────
if __name__ == "__main__":
    sample_chunks = [
        "Apple Inc. was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne. "
        "The company is headquartered in Cupertino, California. Apple designs and sells "
        "consumer electronics, software, and online services.",

        "The iPhone was first introduced by Steve Jobs on January 9, 2007. "
        "It revolutionized the smartphone industry by combining a phone, an iPod, "
        "and an internet communicator into one device.",
    ]

    questions = generate_suggested_questions(sample_chunks)
    print("\nGenerated Questions:")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")
