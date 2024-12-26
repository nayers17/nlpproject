#!/usr/bin/env python

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer

def get_model_probs(model, tokenizer, text):
    """
    Tokenize the input text and get softmax probabilities from the model.
    Returns (predicted_class_index, predicted_class_probability).
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits -> probabilities
    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy().flatten()
    predicted_class = int(np.argmax(probs))
    predicted_prob = probs[predicted_class]
    return predicted_class, predicted_prob

def main():
    model_path = "./5class/results-distilbert-continue/final"  # Adjust to your model folder

    # 1) Load your fine-tuned 5-class model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # 2) Some text to interpret
    text = "The food was decent, but the service was awful."

    # First, get predicted class & prob from a direct model call
    predicted_class, predicted_prob = get_model_probs(model, tokenizer, text)

    # 3) Create the explainer
    cls_explainer = SequenceClassificationExplainer(model=model, tokenizer=tokenizer)

    # 4) Get word attributions
    word_attributions = cls_explainer(text)

    print(f"Input text: {text}")
    print(f"Predicted label: {predicted_class} (prob: {predicted_prob:.3f})")

    print("\nToken Attributions (token -> contribution):")
    for word, score in word_attributions:
        # Positive score = pushes rating up for the predicted class
        # Negative score = pushes rating down
        print(f"{word:<15} {score:.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    main()
