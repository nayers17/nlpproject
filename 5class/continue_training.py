#!/usr/bin/env python
import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro'
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Shift labels [1..5] → [0..4]
def shift_label(example):
    example["label"] = example["label"]
    return example

def tokenize_fn(example, tokenizer):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

def main():
    # Path to your previously trained final model
    prev_model_path = "./5class/results-distilbert/final"

    # Load your trained tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(prev_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        prev_model_path
    )

    # Load Yelp Review Full
    raw_dataset = load_dataset("yelp_review_full")
    train_data = raw_dataset["train"].map(shift_label)
    test_data = raw_dataset["test"].map(shift_label)

    tokenized_train = train_data.map(lambda x: tokenize_fn(x, tokenizer), batched=True)
    tokenized_test = test_data.map(lambda x: tokenize_fn(x, tokenizer), batched=True)

    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # We'll do 2 more epochs; adjust as needed
    training_args = TrainingArguments(
        output_dir="./5class/results-distilbert-continue",  # new output folder
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=3e-5,      # same LR as before (adjust if you like)
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=True,               # if your GPU supports mixed precision
        load_best_model_at_end=True,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics
    )

    print("Starting further training from your existing fine-tuned model...")
    trainer.train()

    # Evaluate
    eval_results = trainer.evaluate()
    print("Evaluation results after continued training:")
    print(eval_results)

    # Save the newly updated model
    trainer.save_model("./5class/results-distilbert-continue/final")
    tokenizer.save_pretrained("./5class/results-distilbert-continue/final")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    main()
