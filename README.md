# Fine-Tuning a Hugging Face Transformer Model (DistilBERT)

## Overview
This project demonstrates the fine-tuning of a Hugging Face Transformer model (DistilBERT) for a natural language processing (NLP) classification task. The pipeline leverages PyTorch for training and an NVIDIA L40 GPU for accelerated computation. This repository is designed to showcase practical NLP skills and includes example code, dataset preparation, and training pipeline setup.

## Repository Structure

NLPPROJECT/
├── 5class/                    # Directory for classification task
│   ├── results-distilbert/        # Results for DistilBERT fine-tuning
│   ├── results-distilbert-continue/ # Continued training results
│   ├── continue_training.py        # Script for continuing training
│   ├── interactive_inference.py    # Script for running inference interactively
│   ├── train.py                    # Main training script
├── originalbinary/            # Directory for other experiments
│   ├── eval.py                    # Evaluation script
│   ├── train.py                   # Training script for another model
│   ├── results-distilroberta/     # Results for DistilRoBERTa model
│       ├── checkpoint-500         # Checkpoint at step 500
│       ├── checkpoint-1000        # Checkpoint at step 1000
│       ├── checkpoint-1250        # Checkpoint at step 1250
├── venv/                      # Python virtual environment
├── .gitignore                 # Ignored files for Git
├── evalmetrics.txt            # Evaluation metrics
├── README.md                  # Project documentation (this file)
├── requirements.txt           # Python dependencies


## Features
- **Fine-Tuning DistilBERT**: Demonstrates how to fine-tune a pre-trained transformer model using Hugging Face and PyTorch.
- **Training Continuation**: Allows resuming training from saved checkpoints.
- **Interactive Inference**: Provides a script to perform inference interactively.
- **Evaluation Metrics**: Logs evaluation metrics for performance monitoring.
- **GPU Optimization**: Leverages mixed-precision training with `fp16` to optimize GPU performance.

## Virtual Environment

bash:
python3 -m venv venv
source venv/bin/activate  

## Requirements
Install the required Python packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## My Huggingface with my current models:

https://huggingface.co/superchillbasedpogger

## How to Use

### 1. Setting Up the Environment
1. Clone this repository:
    ```bash
    git clone <repository_url>
    cd NLPPROJECT
    ```
2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 2. Training the Model
Run the training script for the DistilBERT model:
```bash
python 5class/train.py
```
This script will fine-tune DistilBERT on the classification task and save checkpoints in the `results-distilbert` folder.

### 3. Continuing Training
To resume training from a previous checkpoint, use the `continue_training.py` script:
```bash
python 5class/continue_training.py
```
Checkpoints will be saved in the `results-distilbert-continue` folder.

### 4. Running Inference
Use the `interactive_inference.py` script to interact with the fine-tuned model:
```bash
python 5class/interactive_inference.py
```
Follow the prompts to input text for classification.

### 5. Evaluating the Model
Run the evaluation script to compute metrics:
```bash
python originalbinary/eval.py
```
Metrics will be logged in `evalmetrics.txt`.

## Training Arguments
Key training arguments used in the scripts:
- **Batch Size**: `16` for both training and evaluation.
- **Learning Rate**: `3e-5` with a warm-up ratio of `0.1`.
- **Epochs**: Adjustable; default is `2` for continued training.
- **Mixed Precision**: Enabled with `fp16=True` for better performance.
- **Checkpoints**: Saved at the end of each epoch, with a limit on the total number of checkpoints to retain.

## Results
Results are saved in:
- `5class/results-distilbert` for the initial training.
- `5class/results-distilbert-continue` for continued training.
- Checkpoints are stored periodically for reproducibility and experimentation.

## Hardware Used
This project utilizes the NVIDIA L40 GPU for accelerated training. Mixed-precision training (`fp16`) ensures efficient use of GPU memory and faster computations.

## Acknowledgments
- **Hugging Face Transformers**: For pre-trained models and APIs.
- **PyTorch**: For model training and optimization.

For further details or inquiries, feel free to reach out!
