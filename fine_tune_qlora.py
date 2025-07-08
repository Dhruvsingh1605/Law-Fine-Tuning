# ================================
# File: fine_tune_qlora.py (with trainable check & grad fix)
# ================================
# Prerequisites: huggingface-cli login, accepted HF model terms

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, default_data_collator, BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType

# 1. Configuration
DATA_PATH = "law_qa_formatted.jsonl"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
OUTPUT_DIR = "qlora-mistral-law"
LOGGING_DIR = "./logs"
MAX_LENGTH = 2048
AUTH_TOKEN = True

# 2. Load dataset
dataset = load_dataset("json", data_files={"train": DATA_PATH})["train"]

# 3. Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    use_auth_token=AUTH_TOKEN
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4. Setup quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 5. Load model with quantization
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    use_auth_token=AUTH_TOKEN
)
model.config.pad_token_id = tokenizer.pad_token_id
model.gradient_checkpointing_enable()

# 6. Apply LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)

# 7. Check if any parameters are trainable
trainable = [name for name, param in model.named_parameters() if param.requires_grad]
if not trainable:
    raise RuntimeError("No parameters require gradients! LoRA may not be correctly applied.")

print("\nTrainable parameters:")
for name in trainable:
    print(" -", name)

# 8. Preprocessing

def preprocess_function(examples):
    prompts = examples.get("prompt", [])
    responses = examples.get("response", [])
    texts = [p + r for p, r in zip(prompts, responses)]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)

# 9. Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_dir=LOGGING_DIR,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=2,
    report_to=["tensorboard"],
    fp16=True
)

# 10. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator
)

# 11. Train
trainer.train()

# 12. Save model
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")