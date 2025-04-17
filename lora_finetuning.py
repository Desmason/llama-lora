"""
lora_finetuning.py

This script fine-tunes a LLaMA 1B model using LoRA on the SQuAD v2 dataset.
The script:
  - Loads a pre-trained LLaMA model and its tokenizer.
  - Prepares the model for k-bit training (if applicable) and wraps it with LoRA adapters.
  - Loads and formats the SQuAD v2 dataset.
  - Tokenizes the data in a prompt/answer format suitable for causal language modeling.
  - Sets training hyperparameters via Hugging Face's Trainer and TrainingArguments.
  - Fine-tunes the model and saves the LoRA adapter to the folder "lora-adapter".
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset




# Change this to your actual model repo or local path for LLaMA 1B
base_model_name = "meta-llama/Llama-3.2-1B"

# 1. Load tokenizer and model in FP16 for efficiency
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

# 2. Prepare the model for low-bit training and apply LoRA configuration
# (If using quantization like 4-bit, ensure bitsandbytes is installed and adjust accordingly)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,                     # LoRA rank
    lora_alpha=16,           # Scaling factor for LoRA updates
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to these modules
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

print("Trainable parameters:")
model.print_trainable_parameters()

# 3. Load and format the SQuAD v2 dataset for causal LM training
dataset = load_dataset("rajpurkar/squad_v2")

# Filter out examples with no answer (SQuAD v2 has unanswerable questions)
dataset = dataset.filter(lambda x: len(x["answers"]["text"]) > 0)

def format_example(example):
    question = example["question"]
    context = example["context"]
    answer = example["answers"]["text"][0]
    prompt_text = f"Question: {question}\nContext: {context}\nAnswer:"
    full_text = f"{prompt_text} {answer}"
    return {"text": full_text, "prompt": prompt_text, "answer": answer}

dataset = dataset.map(format_example)

def tokenize_function(example):
    # Tokenizes the formatted text
    tokens = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()  # For causal LM, labels are the input IDs
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# 4. Define training arguments and instantiate the Trainer
training_args = TrainingArguments(
    output_dir="./llama-lora-squad",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].select(range(8000)),
    eval_dataset=tokenized_dataset["validation"].select(range(500)),  # Use a subset for evaluation
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# 5. Fine-tune the model
trainer.train()

# 6. Save the LoRA adapter locally
model.save_pretrained("lora-adapter")
tokenizer.save_pretrained("lora-adapter")
