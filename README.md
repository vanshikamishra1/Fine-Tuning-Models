# Fine-Tuning-Models

---
## Introduction
This project explores and compares four different methods for improving a text response model using a custom set of question-and-answer pairs. The base model, distilgpt2, is adjusted using four approaches: Full Parameter Update, LoRA, QLoRA, and BitFit. Each method is applied to the same conversational dataset, and their results are evaluated using standard measures such as accuracy and perplexity. The goal is to identify which method offers the best balance of performance, speed, and resource use for adapting a language model to new question-and-answer tasks. This document details the process, features of each method, and a comprehensive comparison of their outcomes.

## Setup

1. **Clone the repository and move into the folder:**
   ```sh
   git clone <your-repo-url>
   cd FineTuning
   ```

2. **Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install all required packages:**
   ```sh
   pip install torch transformers datasets peft matplotlib scikit-learn
   ```

---

## Data Preparation

- Generate a dataset of question-and-answer pairs:
  ```sh
  python generate_dataset.py
  ```
- Format the dataset for training:
  ```sh
  python format_dataset.py
  ```
- Split the dataset into training and evaluation sets:
  ```sh
  python split_dataset.py
  ```

---

## Running the Methods

- **Full Parameter Update:**
  ```sh
  python finetune.py
  ```
- **LoRA:**
  ```sh
  python finetune_lora.py
  ```
- **QLoRA:**
  ```sh
  python finetune_qlora.py
  ```
- **BitFit:**
  ```sh
  python finetune_bitfit.py
  ```

Each script will save the updated model and a plot of training metrics.

---

## Features of Each Method

- **Full Parameter Update:**  
  All adjustable parts of the model are changed. Most thorough, but slowest and uses the most memory.

- **LoRA:**  
  Only a small, specific part of the model is changed. Fast and efficient, with high accuracy.

- **QLoRA:**  
  A memory-saving version of LoRA. Useful for limited resources or larger models.

- **BitFit:**  
  Only the bias terms are changed. Fastest and most lightweight, but usually less effective.

---

## Results

| Method   | Perplexity (eval) | Accuracy (eval %) | Training Loss | Training Time (min) | Comments                        |
|----------|-------------------|-------------------|---------------|---------------------|----------------------------------|
| Full     | 1.17              | 15.50             | 0.23          | ~104                | Best perplexity, stable loss     |
| LoRA     | 2.65              | 17.00             | 1.56          | ~71                 | Best accuracy, efficient         |
| QLoRA    | 2.67              | 12.00             | 1.56          | ~72                 | Close to LoRA, efficient         |
| BitFit   | 3.28              | 0.00              | 1.71          | ~73                 | Fastest, but poor accuracy       |

Plots for each method are saved as PNG files in the project folder.

---

## Recommendation

LoRA is recommended for this type of conversational data and resource constraints, as it provides the best balance of accuracy, speed, and memory use.

---
