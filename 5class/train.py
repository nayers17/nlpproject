#!/usr/bin/env python
import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    # Uncomment if you want early stopping
    EarlyStoppingCallback
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

def shift_labels(example):
    """
    Yelp Review Full dataset has labels in [1..5].
    Shift to [0..4] for a 5-class DistilBERT model.
    """
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
    # Print the GPU name for sanity check
    if torch.cuda.is_available():
        print("Using device:", torch.cuda.get_device_name(0))
    else:
        print("Using device: CPU")

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize DistilBERT for 5-class classification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=5
    )

    # Load the entire yelp_review_full dataset
    raw_dataset = load_dataset("yelp_review_full")

    # SHIFT LABELS from [1..5] → [0..4]
    train_data = raw_dataset["train"].map(shift_labels)
    test_data = raw_dataset["test"].map(shift_labels)

    # Debug: Print unique labels to confirm they are {0,1,2,3,4}
    print("Unique train labels:", set(train_data["label"]))
    print("Unique test labels:", set(test_data["label"]))

    # Tokenize
    tokenized_train = train_data.map(
        lambda x: tokenize_fn(x, tokenizer),
        batched=True
    )
    tokenized_test = test_data.map(
        lambda x: tokenize_fn(x, tokenizer),
        batched=True
    )

    # Convert to PyTorch format
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./5class/results-distilbert",
        eval_strategy="epoch",       # Evaluate every epoch
        save_strategy="epoch",       # Save checkpoint every epoch
        num_train_epochs=3,          # Increase to 4–5 if you can wait longer
        per_device_train_batch_size=16,  # Adjust if GPU memory is available (32 if you can handle it)
        per_device_eval_batch_size=16,
        learning_rate=3e-5,
        weight_decay=0.01,           # Common regularization
        warmup_ratio=0.1,            # 10% steps for LR warmup
        fp16=True,                   # Mixed precision on L40
        load_best_model_at_end=True,
        save_total_limit=1,          # Keep only 1 best checkpoint
        logging_steps=100,           # Log training info every 100 steps
        report_to="none"             # or "tensorboard"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
        # Uncomment to use early stopping if you want:
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
    )

    # Train
    trainer.train()

    # Evaluate final
    eval_results = trainer.evaluate()
    print("Eval:", eval_results)

    # Save the final model
    trainer.save_model("./5class/results-distilbert/final")
    tokenizer.save_pretrained("./5class/results-distilbert/final")

if __name__ == "__main__":
    main()
