from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

# Load tokenizer and model
model_name = "./finetuned_distilbert"  # Path to your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load the dataset
dataset = load_dataset("yelp_polarity", split="test[:10%]")  # Replace with your evaluation split

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Define training arguments (no training, just evaluation)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=16,
    report_to="none"  # Disable logging to external services
)

# Create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_dataset,
    compute_metrics=compute_metrics
)

# Evaluate the model
evaluation_results = trainer.evaluate()

# Print the metrics
print(evaluation_results)
