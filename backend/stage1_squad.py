"""
Stage 1: Train DistilBERT on SQuAD 1.1
Teaches the model basic extractive question answering.
Output: models/stage1-squad-distilbert/
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
MODEL_NAME   = "distilbert-base-uncased"
OUTPUT_DIR   = "./stage1-squad-distilbert"
EPOCHS       = 2
BATCH_SIZE   = 16
LR           = 2e-5
MAX_LENGTH   = 384
DOC_STRIDE   = 128
WARMUP_STEPS = 500
# ────────────────────────────────────────────────────────


def preprocess(examples, tokenizer):
    """Tokenize and compute start/end token positions for each answer."""
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

    sample_map    = tokenized.pop("overflow_to_sample_mapping")
    offset_map    = tokenized.pop("offset_mapping")
    start_positions = []
    end_positions   = []

    for i, offsets in enumerate(offset_map):
        sample_idx = sample_map[i]
        answers    = examples["answers"][sample_idx]
        input_ids  = tokenized["input_ids"][i]

        # CLS token position is always 0 — used as fallback (unanswerable)
        cls_idx = input_ids.index(tokenizer.cls_token_id)

        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_idx)
            end_positions.append(cls_idx)
        else:
            char_start = answers["answer_start"][0]
            char_end   = char_start + len(answers["text"][0])

            # Find which tokens cover the answer span
            seq_ids = tokenized.sequence_ids(i)
            # Context tokens have sequence id = 1
            ctx_start = next(j for j, s in enumerate(seq_ids) if s == 1)
            ctx_end   = len(seq_ids) - 1 - next(
                j for j, s in enumerate(reversed(seq_ids)) if s == 1
            )

            # If answer is outside this window, point to CLS
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
    print("=== Stage 1: DistilBERT on SQuAD ===\n")

    # 1. Load tokenizer + model
    print("Loading tokenizer and model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model     = DistilBertForQuestionAnswering.from_pretrained(MODEL_NAME)

    # 2. Load SQuAD 1.1
    print("Loading SQuAD dataset...")
    dataset = load_dataset("squad")
    train_data = dataset["train"]
    val_data   = dataset["validation"]

    # 3. Tokenize
    print("Tokenizing...")
    fn = lambda ex: preprocess(ex, tokenizer)
    train_tok = train_data.map(fn, batched=True, remove_columns=train_data.column_names)
    val_tok   = val_data.map(fn,   batched=True, remove_columns=val_data.column_names)

    train_tok.set_format("torch")
    val_tok.set_format("torch")

    # 4. Training args
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
        model          = model,
        args           = args,
        train_dataset  = train_tok,
        eval_dataset   = val_tok,
        data_collator  = DefaultDataCollator(),
    )

    print(f"\nTraining on {'GPU' if torch.cuda.is_available() else 'CPU'}...")
    trainer.train()

    # 6. Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    import shutil
    shutil.copytree(
        OUTPUT_DIR,
        "/content/drive/MyDrive/DocQuery/models/stage1-squad-distilbert",
        dirs_exist_ok=True
    )
    print(f"\n✅ Stage 1 complete — model saved to Drive")


if __name__ == "__main__":
    main()