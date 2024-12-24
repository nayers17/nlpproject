import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# Define compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

def main():
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained("./finetuned_distilbert")

    # Load a small subset for quick demonstration
    dataset = load_dataset("yelp_polarity", split={
        "train": "train[:1%]", 
        "test": "test[:1%]"
    })

    # Tokenize data
    tokenized_train = dataset["train"].map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_test = dataset["test"].map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Set format for PyTorch
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Training configuration
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,  # increase to 3+ for better results
        logging_steps=10,
        report_to="none",
        load_best_model_at_end=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics
    )

    # Train
    # trainer.train()

    # Save final model
    trainer.save_model("./finetuned_distilbert")
    tokenizer.save_pretrained("./finetuned_distilbert")

    evaluation_results = trainer.evaluate()
    print(evaluation_results)

if __name__ == "__main__":
    # Ensure we run on GPU if available (e.g., your L40)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()

