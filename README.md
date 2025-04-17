
# 🧠 LoRA Fine-Tuning for LLaMA 1B on SQuAD v2

This project demonstrates how to fine-tune the LLaMA 1B model using **Low-Rank Adaptation (LoRA)** on the **SQuAD v2** dataset and evaluate it using **BLEU**, **F1**, and **BERTScore** metrics.

---

## 📂 Project Structure

```
├── lora_finetuning.py              # Fine-tunes LLaMA-1B with LoRA
├── lora_finetuning_evaluation.py  # Evaluates base vs fine-tuned model
├── lora-adapter/                   # Output folder for LoRA adapter weights
├── requirements.txt                # Python dependencies
└── README.md                       # Project instructions
```

---

## 🔧 Setup Instructions

1. **Create a virtual environment (Windows)**

```bash
python -m venv lora_env
.\lora_env\Scripts\activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Log in to Hugging Face (required for LLaMA)**

```bash
huggingface-cli login
```


---

## 🚀 Fine-Tune the Model

```bash
python lora_finetuning.py
```

This will:
- Load the LLaMA 1B base model
- Format and tokenize SQuAD v2
- Apply LoRA adapters
- Train and save LoRA weights to `./lora-adapter/`

---

## 🔍 Evaluate the Model

```bash
python lora_finetuning_evaluation.py
```

This compares:
- **Before Fine-Tuning** (base model)
- **After Fine-Tuning** (with LoRA)
- Metrics: BLEU, token-level F1, BERTScore
- Prints 3 sample outputs from validation

---

## ⏱ Adjusting Number of Epochs

Edit the `num_train_epochs` parameter in `lora_finetuning.py`:

```python
training_args = TrainingArguments(
    ...
    num_train_epochs=2,  # ← Change this
    ...
)
```

---

## ✍️ Author

Yun-Hao Lee  
UT Dallas | MS in Computer Science  
