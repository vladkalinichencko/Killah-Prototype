import os
import json
from datetime import datetime
import math

# Suppress HuggingFace tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dataclasses import dataclass, field
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset, Dataset as HFDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    SchedulerType
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import wandb

# ---------------------------
#  CONFIGURATION BLOCK
# ---------------------------
@dataclass
class TrainingConfig:
    # Models & Paths
    llm_model: str = "google/gemma-3-4b-it"
    output_dir: str = "checkpoints_simplified_lm"
    jsonl_path: str = "transcripts.jsonl"

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"])

    # Training parameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    epochs: int = 3
    lr: float = 2e-4
    warmup_ratio: float = 0.1
    val_test_size: float = 0.05
    clip_threshold: float = 1.0
    
    # Checkpointing & Logging
    resume: bool = True
    val_every_n_steps: int = 200
    save_every_n_steps: int = 200
    log_every_n_steps: int = 5
    
    # Quantization
    quantize_4bit: bool = True
    bnb_compute_dtype: str = "bfloat16"
    
    # W&B
    wandb_project: str = "simplified-lm-finetune"
    wandb_name: str = "gemma-it-finetune"
    wandb_api_key: str = ""
    hf_token: str = ""

# ----------------------------
# 1.  Custom Text Dataset
# ----------------------------
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        
        # Format for instruction tuning
        formatted_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True
        )

        tokenized = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # input_ids and labels are the same for language modeling
        return {
            "input_ids": tokenized.input_ids.squeeze(0),
            "attention_mask": tokenized.attention_mask.squeeze(0)
        }

def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    
    # For Causal LM, labels are the same as input_ids, with padding ignored
    labels = input_ids.clone()
    labels[labels == 0] = -100 # Ignore padding in loss calculation

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# ----------------------------
# 2.  Evaluation loop
# ----------------------------
@torch.no_grad()
def evaluate(model, loader, device, dtype):
    model.eval()
    total_loss = 0
    total_samples = 0
    
    pbar = tqdm(loader, desc="Evaluating")
    
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with autocast(device_type="cuda", dtype=dtype, enabled=torch.cuda.is_available()):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

    avg_loss = total_loss / total_samples
    perplexity = math.exp(avg_loss)
    
    model.train()
    return avg_loss, perplexity

# ----------------------------
# 3.  Training loop
# ----------------------------
def train(config: TrainingConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, config.bnb_compute_dtype) if torch.cuda.is_available() else torch.float32

    os.makedirs(config.output_dir, exist_ok=True)

    # --- 1. Models & Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model, token=config.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # --- LLM Loading with optional 4-bit quantization ---
    if config.quantize_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.llm_model,
            device_map="auto",
            quantization_config=bnb_cfg,
            token=config.hf_token
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.llm_model,
            torch_dtype=dtype,
            device_map={"": device},
            token=config.hf_token
        )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # --- 2. LoRA ---
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights="gaussian",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 3. Dataset & Dataloader ---
    try:
        with open(config.jsonl_path, 'r') as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: JSONL file not found at {config.jsonl_path}")
        return
    
    train_data, val_data = train_test_split(data, test_size=config.val_test_size, random_state=42)
    print(f"Data split: {len(train_data)} training, {len(val_data)} validation.")

    train_dataset = TextDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

    val_dataset = TextDataset(val_data, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    # --- 4. Optimizer, Scheduler, Scaler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    
    num_training_steps = len(train_loader) * config.epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / num_warmup_steps))

    scaler = GradScaler(enabled=torch.cuda.is_available())

    # --- 5. Training Loop ---
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast(device_type="cuda", dtype=dtype, enabled=torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at step {global_step}. Skipping step.")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_threshold)
            scaler.step(optimizer)
            scaler.update()
            
            if global_step < num_warmup_steps:
                scheduler.step()
            
            global_step += 1
            
            if global_step % config.log_every_n_steps == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})
                if wandb.run:
                    wandb.log({
                        "train/loss": loss.item(), 
                        "learning_rate": optimizer.param_groups[0]['lr'], 
                        "step": global_step
                    })

            # --- Validation ---
            if global_step % config.val_every_n_steps == 0:
                val_loss, perplexity = evaluate(model, val_loader, device, dtype)
                pbar.write(f"Step {global_step} | Validation Loss: {val_loss:.4f} | Perplexity: {perplexity:.2f}")
                if wandb.run:
                    wandb.log({"val/loss": val_loss, "val/perplexity": perplexity, "step": global_step})
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.save_pretrained(os.path.join(config.output_dir, "best_model"))
                    pbar.write(f"--- New Best Model Saved ---")

    print("Training finished.")
    model.save_pretrained(os.path.join(config.output_dir, "final_model"))
    print("Final model saved.")

# ----------------------------
# 6.  Entry-point
# ----------------------------
if __name__ == "__main__":
    config = TrainingConfig()

    if config.wandb_api_key:
        try:
            wandb.login(key=config.wandb_api_key)
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_name,
                resume="allow",
                config=vars(config),
            )
            wandb.define_metric("train/loss", step_metric="step")
            wandb.define_metric("val/loss", step_metric="step")
        except Exception as e:
            print(f"WandB login or init failed: {e}. Running in disabled mode.")
            wandb.init(mode="disabled")
    else:
        print("WANDB_API_KEY not found. Running wandb in disabled mode.")
        wandb.init(mode="disabled")
    
    train(config) 