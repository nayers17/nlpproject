#!/usr/bin/env python
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_prediction(model, tokenizer, text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
    predicted_class = np.argmax(probs).item()

    return predicted_class, probs

def main():
    # Path to your fine-tuned model folder (which has config.json, pytorch_model.bin, etc.)
    model_path = "./5class/results-distilbert/final"

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    print("=== Interactive Inference ===")
    print("Enter some text to get a predicted sentiment rating [0..4]. Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter text: ")
        if user_input.lower() == "exit":
            break
        
        pred_class, probabilities = get_prediction(model, tokenizer, user_input)
        print(f"Predicted class (0..4): {pred_class}")
        print(f"Probabilities: {probabilities}\n")

if __name__ == "__main__":
    main()
