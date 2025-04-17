import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import evaluate
import string, re

# Set your base model name and adapter path
base_model_name = "meta-llama/Llama-3.2-1B"
adapter_path = "./lora-adapter"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16
).to("cuda")

# Load fine-tuned LoRA model
lora_model = PeftModel.from_pretrained(base_model, adapter_path).to("cuda")

# Load and prepare SQuAD v2 dataset
dataset = load_dataset("squad_v2")
dataset = dataset.filter(lambda x: len(x["answers"]["text"]) > 0)

def format_example(example):
    question = example["question"]
    context = example["context"]
    answer = example["answers"]["text"][0]
    prompt_text = f"Question: {question}\nContext: {context}\nAnswer:"
    full_text = f"{prompt_text} {answer}"
    return {"text": full_text, "prompt": prompt_text, "answer": answer}

dataset = dataset.map(format_example)

# Utility functions
def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(f"[{re.escape(string.punctuation)}]", "", s)
    return s.strip()

def compute_f1(pred: str, truth: str) -> float:
    pred_tokens = normalize_text(pred).split()
    true_tokens = normalize_text(truth).split()
    if not pred_tokens or not true_tokens:
        return 0.0
    common = set(pred_tokens) & set(true_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(true_tokens)
    return 2 * (precision * recall) / (precision + recall)

# Load evaluation metrics
bleu_metric = evaluate.load("bleu")
bertscore_metric = evaluate.load("bertscore")

# Evaluation function
def evaluate_model(model, tokenizer, dataset, num_samples=100, label=""):
    model.eval()
    preds, refs, f1s = [], [], []

    if label:
        print(f"\n--- {label} Sample Outputs ---")

    for i in range(num_samples):
        prompt = dataset["validation"][i]["prompt"]
        true_answer = dataset["validation"][i]["answer"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_answer = output_text[len(prompt):].strip()

        preds.append(pred_answer)
        refs.append([true_answer])
        f1s.append(compute_f1(pred_answer, true_answer))

        if i < 3 and label:
            print(f"Q: {prompt}\nA (true): {true_answer}\nA (pred): {pred_answer}\n")

    bleu = bleu_metric.compute(predictions=preds, references=refs)["bleu"]
    bert_f1 = sum(bertscore_metric.compute(predictions=preds, references=[r[0] for r in refs], lang="en")["f1"]) / num_samples
    avg_f1 = sum(f1s) / num_samples

    print(f"{label} BLEU Score: {bleu:.4f}")
    print(f"{label} F1 Score: {avg_f1 * 100:.2f}%")
    print(f"{label} BERTScore (F1): {bert_f1 * 100:.2f}%\n")

# Run evaluations
evaluate_model(base_model, tokenizer, dataset, label="Before Fine-Tuning")
evaluate_model(lora_model, tokenizer, dataset, label="After Fine-Tuning")
