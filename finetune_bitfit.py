from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, TrainerCallback
)
from datasets import load_dataset
import torch
import math
import re
import matplotlib.pyplot as plt

model_name = "distilgpt2"
data_files = {"train": "train.jsonl", "validation": "eval.jsonl"}
output_dir = "finetuned_model_bitfit"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)


for name, param in model.named_parameters():
    if "bias" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

def preprocess(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

raw_datasets = load_dataset("json", data_files=data_files)
train_ds = raw_datasets["train"].map(preprocess, batched=True)
eval_ds = raw_datasets["validation"].map(preprocess, batched=True)
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10,
    save_total_limit=2,
    logging_steps=5,
    learning_rate=5e-5,
    fp16=False
)

losses, grad_norms, learning_rates, epochs = [], [], [], []

class MetricsLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs and "epoch" in logs:
            losses.append(logs["loss"])
            grad_norms.append(logs.get("grad_norm", 0))
            learning_rates.append(logs.get("learning_rate", 0))
            epochs.append(logs["epoch"])

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[MetricsLoggerCallback()]
)

trainer.train()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("BitFit Fine-tuning complete.")

eval_results = trainer.evaluate()
perplexity = math.exp(eval_results["eval_loss"])
print(f"Perplexity (eval set): {perplexity:.2f}")

def normalize(text):
    return re.sub(r'[^\w\s]', '', text.lower().strip())

def get_predicted_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in decoded:
        return decoded.split("Assistant:")[-1].strip()
    else:
        return decoded.strip()

total, correct = 0, 0
for ex in eval_ds:
    text = tokenizer.decode(ex["input_ids"], skip_special_tokens=True)
    if "User:" in text and "Assistant:" in text:
        user_part = text.split("User:")[1].split("Assistant:")[0].strip()
        assistant_gt = text.split("Assistant:")[1].strip()
        prompt = f"User: {user_part}\nAssistant:"
        pred = get_predicted_answer(prompt)
        if normalize(pred) == normalize(assistant_gt):
            correct += 1
        total += 1

accuracy = (correct / total) * 100 if total > 0 else 0
print(f"Custom Q&A Accuracy (eval set): {accuracy:.2f}%")

plt.figure(figsize=(12, 7))
plt.plot(epochs, losses, marker='o', label='Loss')
plt.plot(epochs, grad_norms, marker='x', label='Grad Norm')
plt.plot(epochs, learning_rates, marker='s', label='Learning Rate')
plt.axhline(y=perplexity, color='r', linestyle='--', label=f'Perplexity (eval): {perplexity:.2f}')
plt.axhline(y=accuracy/10, color='g', linestyle='-.', label=f'Accuracy (eval %): {accuracy:.2f} (scaled)')
plt.xlabel('Epoch')
plt.legend()
plt.title('BitFit Training Metrics + Eval Perplexity & Accuracy')
plt.savefig('training_metrics_bitfit.png')
print("All metrics plotted and saved as training_metrics_bitfit.png")