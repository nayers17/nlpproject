import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

def predict_sentiment(texts, model_name="./finetuned_distilbert"):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Tokenize input texts
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Move inputs to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)

    # Convert logits to labels
    predictions = torch.argmax(probabilities, dim=-1).cpu().numpy()
    return predictions, probabilities.cpu().numpy()

if __name__ == "__main__":
    # Sample texts for testing
    test_texts = [
        "The service was amazing and the food was excellent!",
        "I had a terrible experience, the staff was rude, and the food was cold.",
        "The product works as expected, no complaints.",
        "Worst purchase ever. It broke within a day.",
        "Absolutely loved it! Highly recommend."
    ]

    # Predict sentiments
    predictions, probabilities = predict_sentiment(test_texts)

    # Display results
    for text, prediction, probability in zip(test_texts, predictions, probabilities):
        sentiment = "Positive" if prediction == 1 else "Negative"
        print(f"Text: {text}\nSentiment: {sentiment} (Confidence: {np.max(probability):.2f})\n")
