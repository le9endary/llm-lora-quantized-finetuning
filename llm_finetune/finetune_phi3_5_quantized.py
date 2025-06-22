import os
import json
import random
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login

# Loading the united development program data
with open('un_1.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

text_data = [item['p_message'] for item in data if 'p_message' in item]
combined_text_data = [' '.join(text.splitlines()) for text in text_data]
sampled_data = combined_text_data[:4000]

dataset = Dataset.from_dict({"text": sampled_data})
base_model = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model)

def preprocess_func(examples):
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding="max_length",
        max_length=2048,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(preprocess_func, batched=True)

# Quantized 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Apply LoRA for Fine-tuning
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["gate_up_proj", "down_proj", "qkv_proj", "o_proj"],
)

model = get_peft_model(model, peft_config)

# Training Config
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    logging_dir="./logs",
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    evaluation_strategy="no",
    learning_rate=1e-4,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# Start Fine-tuning
trainer.train()

# Pushing to Hugging Face, add your login
login(token="")

model.push_to_hub("Kn1ght0/phi-3.5-2epoch-UN-4000")
tokenizer.push_to_hub("Kn1ght0/phi-3.5-2epoch-UN-4000")
