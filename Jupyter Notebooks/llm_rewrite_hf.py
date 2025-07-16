import os
import json
from datetime import datetime
import math
import zipfile
import io
import pandas as pd
from datasets import load_dataset
import csv

# Suppress HuggingFace tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dataclasses import dataclass, field
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn.utils.rnn import pad_sequence
import wandb
from sklearn.model_selection import train_test_split
import random

# ---------------------------
#  CONFIGURATION BLOCK
# ---------------------------
@dataclass
class TrainingConfig:
    # Models & Paths
    llm_model: str = "google/gemma-3-4b-pt"
    output_dir: str = "checkpoints_llm_grandmaster"
    resume_from_dir: str = "checkpoints_llm_coedit_base"  # Directory to load checkpoints from
    dataset_name: str = "Vikhrmodels/GrandMaster-PRO-MAX"

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"])
    init_lora_weights: bool = True

    # Training parameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    epochs: int = 3
    lr: float = 2e-5
    warmup_steps: int = 100
    val_test_size: float = 0.01
    clip_threshold: float = 5.0
    
    # Scheduler
    cosine_restart_steps: int = 250
    val_subset_fraction: float = 1.0 # Fraction of validation data to use (1.0 for all)

    # Checkpointing & Logging
    resume: bool = True
    val_every_n_steps: int = 500
    save_every_n_steps: int = 50
    log_every_n_steps: int = 5
    
    # Generation
    max_new_tokens: int = 70
    beam_size: int = 4

    # W&B
    wandb_project: str = "llm-coedit"
    wandb_name: str = "grandmaster-run"
    wandb_api_key: str = ""

    # Quantization
    quantize_4bit: bool = True
    bnb_compute_dtype: str = "bfloat16"
    hf_token: str = ""

    # Debug flag
    debug_overfit: bool = False

# ----------------------------
#  Custom Dataset
# ----------------------------
class TextDataset(Dataset):
    """Dataset based on HF dataset with 'conversation' structure."""
    def __init__(self, data_list, tokenizer):
        self.data = data_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        instruction = ""
        target_text = ""

        conversation = item.get("conversation")
        if conversation and isinstance(conversation, list):
            for message in conversation:
                if message.get("role") == "user":
                    instruction = message.get("content", "").strip()
                elif message.get("role") == "assistant":
                    target_text = message.get("content", "").strip()

        if instruction:
            prompt = f"{instruction}\n\n"
        else:
            prompt = "" # For continuation tasks

        full_text = prompt + target_text

        full_ids = self.tokenizer(full_text, return_tensors="pt", max_length=512, truncation=True).input_ids.squeeze(0)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
        prompt_len = prompt_ids.size(0)

        return {"input_ids": full_ids, "prompt_len": prompt_len}

def collate_fn(batch, tokenizer):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = input_ids.clone()
    for i, prompt_len in enumerate([item['prompt_len'] for item in batch]):
        labels[i, :prompt_len] = -100
    return {"input_ids": input_ids, "labels": labels}

# ----------------------------
#  Evaluation loop
# ----------------------------
@torch.no_grad()
def evaluate(model, loader, tokenizer, device, dtype, config):
    model.eval()
    
    total_loss = 0
    total_samples = 0
    
    pbar = tqdm(loader, desc="Evaluating")
    
    example_outputs = []

    for batch in pbar:
        if batch is None: continue
        
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        with autocast(device_type="cuda", dtype=dtype, enabled=torch.cuda.is_available()):
            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

            # Generation: generate continuation after prompt
            prompt_len = (labels[0] == -100).nonzero(as_tuple=True)[0][-1].item() + 1
            prompt_ids = input_ids[0][:prompt_len]
            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            generated_ids = model.generate(
                input_ids=prompt_ids.unsqueeze(0),
                max_new_tokens=config.max_new_tokens,
                num_beams=config.beam_size,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen_text = tokenizer.decode(generated_ids[0][prompt_ids.size(0):], skip_special_tokens=True)
            tgt_text = tokenizer.decode(input_ids[0][prompt_len:], skip_special_tokens=True)
            if len(example_outputs) < 5:
                example_outputs.append(f"\n  PROMPT    : {prompt_text.strip()}\n  TARGET    : {tgt_text}\n  GENERATED : {gen_text}\n")

    pbar.write("\n--- Validation Examples ---")
    for example in example_outputs:
        pbar.write(example)
    pbar.write("-------------------------\n")
    
    model.train()
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    
    return avg_loss

# ----------------------------
#  Training loop
# ----------------------------
def train(config: TrainingConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    os.makedirs(config.output_dir, exist_ok=True)

    # --- 1. Models & Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model, padding_side="left", token=config.hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})

    # Trim vocab size as in reference notebook (262_208)
    cfg = AutoConfig.from_pretrained(config.llm_model, token=config.hf_token)
    cfg.vocab_size = 262208

    # --- LLM Loading with optional 4-bit quantization ---
    if config.quantize_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_compute_dtype),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model,
            config=cfg,
            device_map="auto",
            torch_dtype=getattr(torch, config.bnb_compute_dtype),
            quantization_config=bnb_cfg,
            trust_remote_code=True,
            token=config.hf_token if config.hf_token else None,
            attn_implementation="eager",
        )
        llm.gradient_checkpointing_enable()
        llm = prepare_model_for_kbit_training(llm)
    else:
        llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model,
            config=cfg,
            torch_dtype=dtype,
            device_map={"": device},
            attn_implementation="eager",
            trust_remote_code=True,
        )
    # Resize embeddings only if tokenizer length differs
    if llm.get_input_embeddings().weight.shape[0] != len(tokenizer):
        llm.resize_token_embeddings(len(tokenizer))

    # --- 2. LoRA ---
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights=config.init_lora_weights,
    )
    llm = get_peft_model(llm, lora_config)
    llm.print_trainable_parameters()

    # --- 3. Dataset & Dataloader ---
    try:
        print(f"✅ Loading dataset '{config.dataset_name}' from Hugging Face Hub...")
        dataset = load_dataset(config.dataset_name, split="train", token=config.hf_token)
        
        print(f"✅ Splitting dataset ({1-config.val_test_size:.0%} train, {config.val_test_size:.0%} val)...")
        dataset_splits = dataset.train_test_split(test_size=config.val_test_size, seed=42)
        
        train_data = list(dataset_splits['train'])
        val_data = list(dataset_splits['test'])

    except Exception as e:
        print(f"❌ An unexpected error occurred while loading or processing the dataset: {e}")
        return

    # Reduce validation set if needed
    if config.val_subset_fraction < 1.0:
        val_subset_size = int(len(val_data) * config.val_subset_fraction)
        val_data = random.sample(val_data, val_subset_size)
        print(f"✅ Using a random subset of validation data: {len(val_data)} samples ({config.val_subset_fraction*100:.1f}%).")

    print(f"Data loaded: {len(train_data)} training, {len(val_data)} validation.")
    if config.debug_overfit:
        train_data = train_data[:1]
        val_data = train_data
        print("Debug overfit mode enabled: using a single training sample.")

    train_dataset = TextDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer), num_workers=0, pin_memory=True)

    val_dataset = TextDataset(val_data, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer), num_workers=0, pin_memory=True)

    # --- 4. optimizer, Scheduler, Scaler ---
    params_to_train = [p for p in llm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_train, lr=config.lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / config.warmup_steps))
    print(f"Using LambdaLR (linear warmup) scheduler with {config.warmup_steps} steps")

    scaler = GradScaler(enabled=torch.cuda.is_available())

    # --- 5. Checkpoint Loading ---
    start_epoch = 0
    start_batch_idx = 0
    global_step = 0
    best_val_loss = float('inf')
    
    # Determine checkpoint path for resumption
    resume_dir = config.resume_from_dir if config.resume_from_dir and os.path.isdir(config.resume_from_dir) else config.output_dir
    checkpoint_path = os.path.join(resume_dir, "latest.pt")

    if os.path.exists(checkpoint_path) and config.resume:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        adapter_path_latest = os.path.join(resume_dir, "latest_adapter")
        if os.path.isdir(adapter_path_latest):
            llm.load_adapter(adapter_path_latest, adapter_name="default", is_trainable=True)
            print(f"Loaded adapter from {adapter_path_latest}.")
        else:
            print(f"Adapter directory {adapter_path_latest} not found. Skipping adapter load.")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        start_batch_idx = checkpoint.get('batch_idx', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Run validation if resuming on a validation step
        if global_step > 0 and global_step % config.val_every_n_steps == 0:
            print(f"\nRunning validation for resumed step {global_step} before continuing training...")
            val_loss = evaluate(llm, val_loader, tokenizer, device, dtype, config)
            print(f"Step {global_step} | Validation Loss: {val_loss:.3f}")
            
            if wandb.run and not wandb.run.disabled:
                wandb.log({"val/loss": val_loss, "step": global_step})
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = os.path.join(config.output_dir, "best.pt")
                torch.save({
                    'epoch': start_epoch,
                    'global_step': global_step,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                }, best_checkpoint_path)
                llm.save_pretrained(os.path.join(config.output_dir, "best_adapter"), token=config.hf_token or None)
                print(f"--- New Best Model ---\nSaved new best model at step {global_step} with val_loss: {best_val_loss:.3f}\n----------------------\n")
        
        print(f"Resumed from Epoch {start_epoch}, Step {global_step}, Batch {start_batch_idx}")

    # --- 6. Training Loop ---
    for epoch in range(start_epoch, config.epochs):
        llm.train()
        
        # Handle resumption
        current_train_data = train_data
        
        if epoch == start_epoch and start_batch_idx > 0 and config.resume:
            samples_to_skip = start_batch_idx * config.batch_size
            if samples_to_skip < len(train_data):
                print(f"Resuming epoch {epoch}. Skipping {samples_to_skip} samples.")
                current_train_data = train_data[samples_to_skip:]
            else:
                print(f"Epoch {epoch} already completed. Starting next epoch.")
                start_batch_idx = 0
                continue
        
        epoch_train_dataset = TextDataset(current_train_data, tokenizer)
        epoch_train_loader = DataLoader(epoch_train_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer), num_workers=0, pin_memory=True)
        
        initial_batch_idx = start_batch_idx if epoch == start_epoch else 0
        total_batches = len(epoch_train_loader) + initial_batch_idx
        pbar = tqdm(epoch_train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", initial=initial_batch_idx, total=total_batches)
        
        for i, batch in enumerate(pbar, start=initial_batch_idx):
            is_update_step = (i + 1) % config.gradient_accumulation_steps == 0
            
            if batch is None: continue
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            with autocast(device_type="cuda", dtype=dtype, enabled=torch.cuda.is_available()):
                outputs = llm(input_ids=input_ids, labels=labels)
                loss = outputs.loss / config.gradient_accumulation_steps

            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at step {global_step}. Skipping step.")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            
            if is_update_step:
                scaler.unscale_(optimizer)
                
                grad_norm_before_clip = 0
                for p in params_to_train:
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        grad_norm_before_clip += param_norm.item() ** 2
                grad_norm_before_clip = (grad_norm_before_clip ** 0.5) if grad_norm_before_clip > 0 else 0.0

                torch.nn.utils.clip_grad_norm_(params_to_train, config.clip_threshold)
                
                grad_norm_after_clip = 0
                for p in params_to_train:
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        grad_norm_after_clip += param_norm.item() ** 2
                grad_norm_after_clip = (grad_norm_after_clip ** 0.5) if grad_norm_after_clip > 0 else 0.0

                scaler.step(optimizer)
                scaler.update()
            
                if global_step < config.warmup_steps:
                    scheduler.step()
            
                global_step += 1
                
                gpu_mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
                pbar.set_postfix({
                    "loss": f"{loss.item()*config.gradient_accumulation_steps:.3f}", 
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "gn": f"{grad_norm_after_clip:.2f}",
                    "mem_gb": f"{gpu_mem_alloc:.2f}",
                    "time": datetime.now().strftime("%H:%M:%S")
                })

                if global_step % config.log_every_n_steps == 0:
                    pbar.write(f"Step {global_step} | Train Loss: {loss.item()*config.gradient_accumulation_steps:.4f}")
                    if wandb.run and not wandb.run.disabled:
                        gpu_mem_max = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                        wandb.log({
                            "train/loss": loss.item()*config.gradient_accumulation_steps,
                            "learning_rate": optimizer.param_groups[0]['lr'],
                            "step": global_step,
                            "grad_norm_before_clip": grad_norm_before_clip,
                            "grad_norm_after_clip": grad_norm_after_clip,
                            "gpu_mem_allocated_gb": gpu_mem_alloc,
                            "gpu_mem_max_allocated_gb": gpu_mem_max,
                            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })

                # --- Checkpointing & Validation ---
                if global_step > 0 and global_step % config.save_every_n_steps == 0:
                    latest_checkpoint_path = os.path.join(config.output_dir, "latest.pt")
                    torch.save({
                        'epoch': epoch, 
                        'batch_idx': i,
                        'global_step': global_step,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'best_val_loss': best_val_loss,
                    }, latest_checkpoint_path)
                    llm.save_pretrained(os.path.join(config.output_dir, "latest_adapter"), token=config.hf_token or None)
                    pbar.write(f"\n--- Checkpoint Saved ---\nSaved latest model at step {global_step}.\n------------------------\n")

                if global_step > 0 and global_step % config.val_every_n_steps == 0:
                    pbar.write(f"\nRunning validation at step {global_step}...")
                    val_loss = evaluate(llm, val_loader, tokenizer, device, dtype, config)
                    pbar.write(f"Step {global_step} | Validation Loss: {val_loss:.3f}")
                    
                    if wandb.run and not wandb.run.disabled:
                        wandb.log({"val/loss": val_loss, "step": global_step})
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_checkpoint_path = os.path.join(config.output_dir, "best.pt")
                        torch.save({
                            'epoch': epoch,
                            'global_step': global_step,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                            'best_val_loss': best_val_loss,
                        }, best_checkpoint_path)
                        llm.save_pretrained(os.path.join(config.output_dir, "best_adapter"), token=config.hf_token or None)
                        pbar.write(f"--- New Best Model ---\nSaved new best model at step {global_step} with val_loss: {best_val_loss:.3f}\n----------------------\n")
        
        # Reset start_batch_idx for next epoch
        start_batch_idx = 0

# ----------------------------
#  Entry-point
# ----------------------------
if __name__ == "__main__":
    config = TrainingConfig()
    
    # Overrides for GrandMaster training
    config.lora_dropout = 0.2
    config.val_subset_fraction = 0.1 # Use 10% of the validation set
    config.gradient_accumulation_steps = 32 // config.batch_size
    config.debug_overfit = False  # Disable for full training

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
            wandb.define_metric("learning_rate", step_metric="step")
        except Exception as e:
            print(f"WandB login or init failed: {e}. Running in disabled mode.")
            wandb.init(mode="disabled")
    else:
        print("WANDB_API_KEY not found. Running wandb in disabled mode.")
        wandb.init(mode="disabled")
    
    train(config) 