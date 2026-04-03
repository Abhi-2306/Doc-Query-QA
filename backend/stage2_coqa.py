"""
Stage 2: Fine-tune DistilBERT (from Stage 1) on CoQA
Teaches the model conversational follow-up question answering.
Input:  models/stage1-squad-distilbert/
Output: models/distilbert-final/
"""

import os
from transformers import set_seed
set_seed(42)

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
from datasets import load_dataset
import torch


# ── Config ──────────────────────────────────────────────
STAGE1_DIR   = "./stage1-squad-distilbert"
OUTPUT_DIR   = "./distilbert-final"
EPOCHS       = 2
BATCH_SIZE   = 32
LR           = 2e-5
MAX_LENGTH   = 256
DOC_STRIDE   = 64
WARMUP_STEPS = 200
# ────────────────────────────────────────────────────────


def flatten_coqa(dataset):
    """
    CoQA stores multiple Q&A turns per story in a single row.
    This flattens it so each (question, answer, context) is its own example.
    """
    rows = {"context": [], "question": [], "answers": []}

    for ex in dataset:
        ctx = ex["story"]
        for q, a_text, a_start in zip(
            ex["questions"],
            ex["answers"]["input_text"],
            ex["answers"]["answer_start"],
        ):
            rows["context"].append(ctx)
            rows["question"].append(q)
            rows["answers"].append({
                "text":         [a_text],
                "answer_start": [a_start],
            })

    from datasets import Dataset
    return Dataset.from_dict(rows)


def preprocess(examples, tokenizer):
    """Same tokenization logic as Stage 1 — maps char positions to token positions."""
    questions = [q.strip() for q in examples["question"]]

    tokenized = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map      = tokenized.pop("overflow_to_sample_mapping")
    offset_map      = tokenized.pop("offset_mapping")
    start_positions = []
    end_positions   = []

    for i, offsets in enumerate(offset_map):
        sample_idx = sample_map[i]
        answers    = examples["answers"][sample_idx]
        input_ids  = tokenized["input_ids"][i]

        cls_idx = input_ids.index(tokenizer.cls_token_id)

        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_idx)
            end_positions.append(cls_idx)
        else:
            char_start = answers["answer_start"][0]
            char_end   = char_start + len(answers["text"][0])

            seq_ids   = tokenized.sequence_ids(i)
            ctx_start = next(j for j, s in enumerate(seq_ids) if s == 1)
            ctx_end   = len(seq_ids) - 1 - next(
                j for j, s in enumerate(reversed(seq_ids)) if s == 1
            )

            if offsets[ctx_start][0] > char_end or offsets[ctx_end][1] < char_start:
                start_positions.append(cls_idx)
                end_positions.append(cls_idx)
            else:
                tok_start = ctx_start
                while tok_start <= ctx_end and offsets[tok_start][0] <= char_start:
                    tok_start += 1
                start_positions.append(tok_start - 1)

                tok_end = ctx_end
                while tok_end >= ctx_start and offsets[tok_end][1] >= char_end:
                    tok_end -= 1
                end_positions.append(tok_end + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"]   = end_positions
    return tokenized


def main():
    print("=== Stage 2: DistilBERT fine-tune on CoQA ===\n")

    # 1. Load tokenizer + Stage 1 model
    print(f"Loading Stage 1 model from {STAGE1_DIR}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(STAGE1_DIR)
    model     = DistilBertForQuestionAnswering.from_pretrained(STAGE1_DIR)

    # 2. Load + flatten CoQA
    print("Loading CoQA dataset...")
    raw = load_dataset("coqa")
    train_flat = flatten_coqa(raw["train"]).select(range(6000))
    val_flat   = flatten_coqa(raw["validation"]).select(range(500))

    train_flat = flatten_coqa(raw["train"])
    val_flat   = flatten_coqa(raw["validation"])

    print(f"Train examples: {len(train_flat)} | Val examples: {len(val_flat)}")

    # 3. Tokenize
    print("Tokenizing...")
    fn        = lambda ex: preprocess(ex, tokenizer)
    train_tok = train_flat.map(fn, batched=True, remove_columns=train_flat.column_names)
    val_tok   = val_flat.map(fn,   batched=True, remove_columns=val_flat.column_names)

    train_tok.set_format("torch")
    val_tok.set_format("torch")

    # 4. Training args — smaller LR, more epochs than Stage 1
    args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        learning_rate               = LR,
        weight_decay                = 0.01,
        warmup_steps                = WARMUP_STEPS,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        logging_steps               = 100,
        fp16                        = torch.cuda.is_available(),
    )

    # 5. Train
    trainer = Trainer(
        model         = model,
        args          = args,
        train_dataset = train_tok,
        eval_dataset  = val_tok,
        data_collator = DefaultDataCollator(),
    )

    print(f"\nFine-tuning on {'GPU' if torch.cuda.is_available() else 'CPU'}...")
    trainer.train()

    # 6. Save final model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    import shutil
    shutil.copytree(
        OUTPUT_DIR,
        "/content/drive/MyDrive/DocQuery/models/distilbert-final",
        dirs_exist_ok=True
    )
    print(f"\n✅ Stage 2 complete — final model saved to Drive")


if __name__ == "__main__":
    main()