#!/usr/bin/env python
import shap
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# A small helper to turn input text into model probabilities
def predict_proba(texts, tokenizer, model, device):
    """
    Returns a Numpy array of shape (num_texts, num_labels)
    with model probabilities.
    """
    # Tokenize
    encoded = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoded)

    # Convert logits -> probabilities
    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
    return probs

def main():
    # 1. Load your fine-tuned DistilBERT from disk
    model_path = "./5class/results-distilbert/final"  # Adjust if needed
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 2. Some example texts to interpret
    example_texts = [
        "The meal was great! Loved the ambience too.",
        "Terrible service. I will not come back again.",
        "Decent food, but they forgot my drink.",
        "Absolutely perfect dinner, highly recommend!"
    ]

    # 3. Wrap a small “predict function” for SHAP
    #    so it only needs the texts, not tokenizers, etc. each time.
    def predict_fn(texts):
        return predict_proba(texts, tokenizer, model, device)

    # 4. Create a KernelExplainer
    #    We'll use a small sample (or the same) as a "background" for SHAP.
    #    For a quick demo, using the first example as background is okay.
    background_texts = example_texts[:1]
    explainer = shap.KernelExplainer(
        model=predict_fn,
        data=background_texts
    )

    # 5. Actually compute SHAP values for all example texts
    #    NOTE: This can be slow if you do many samples or big data.
    shap_values = explainer.shap_values(example_texts, nsamples=100) 
    # "nsamples" controls the number of evaluations; higher = more accurate, slower.

    # 6. Visualize the results
    #    SHAP can produce text plots, HTML, or other plot types.
    #    We’ll do a simple text-based explanation here.

    for i, text in enumerate(example_texts):
        print(f"\n=== Text #{i+1}: {text}")
        # shap_values is a list of arrays (one per output class).
        # For a 5-class model, shap_values has length 5. Let's pick the class
        # that was actually predicted to highlight how that class was influenced.

        predicted_probs = predict_fn([text])[0]
        predicted_label = int(np.argmax(predicted_probs))
        predicted_conf = predicted_probs[predicted_label]

        print(f"Predicted label: {predicted_label} (confidence: {predicted_conf:.3f})")

        # We'll show the token-level attribution for the *predicted* class
        # shap_values[class_idx][sample_idx] is a list of per-token attributions
        # for that class, for the given sample.
        # We'll use shap's built-in text plot (in the console).

        class_explainer_values = shap_values[predicted_label][i]
        tokens = explainer.data[i].split()  # crude splitting of the text
        # Some mismatch can occur if the tokenizer splits differently,
        # but for a quick "wow" factor, let's keep it simple.

        # shap.initjs() only needed if you do HTML display in a notebook
        # Instead we'll do shap.plots:
        shap.plots.text(
            shap.Explanation(values=class_explainer_values, data=tokens)
        )

if __name__ == "__main__":
    main()
