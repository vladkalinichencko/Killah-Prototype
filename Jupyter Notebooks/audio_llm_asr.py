import os
import json
import zipfile
import io
from datetime import datetime

# Suppress HuggingFace tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dataclasses import dataclass, field
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from datasets import load_dataset, Dataset as HFDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    WhisperProcessor,
    WhisperModel,
    WavLMModel,
    AutoProcessor
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import wandb
from sklearn.model_selection import train_test_split
import jiwer
import random

# ---------------------------
#  CONFIGURATION BLOCK
# ---------------------------
@dataclass
class TrainingConfig:
    # Models & Paths
    llm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v0.4"
    audio_encoder_model: str = "openai/whisper-small"
    output_dir: str = "checkpoints_audio_llm"
    jsonl_path: str = "transcripts.jsonl"
    zip_path: str = "LibriSpeech.zip"

    # Projector
    projector_hidden_dim: int = 2048

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    init_lora_weights: bool = True

    # Discretization
    top_k: int = 10
    downsample_k: int = 5

    # Training parameters
    batch_size_stage1: int = 16
    batch_size_stage2: int = 8
    gradient_accumulation_steps: int = 2
    epochs_stage1: int = 10
    epochs_stage2: int = 2
    lr_stage1: float = 2e-5
    lr_stage2: float = 1e-5
    warmup_steps: int = 1000
    val_test_size: float = 0.05
    
    # Checkpointing & Logging
    resume: bool = True
    val_every_n_steps: int = 250
    save_every_n_steps: int = 50
    log_every_n_steps: int = 10
    
    # Generation
    max_new_tokens: int = 70
    beam_size: int = 4

    # W&B
    wandb_project: str = "audio-llm-asr"
    wandb_name: str = "stage1-run-4"
    wandb_api_key: str = "f2c28a0327b6e9b15d2d3a911be9d6cce58fcd39"

# ----------------------------
# 1.  Audio -> LLM Bridge
# ----------------------------

class Projector(nn.Module):
    """Two-layer MLP with ReLU used in the SLAM-ASR paper."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        return self.net(x.view(b * t, d)).view(b, t, -1)

def downsample_and_concat(x: torch.Tensor, k: int) -> torch.Tensor:
    """Concatenate every k consecutive frames."""
    b, t, c = x.shape
    if t == 0:
        return x.view(b, 0, c)
    n = t // k
    x = x[:, :n * k, :].reshape(b, n, k, c)
    return x.reshape(b, n, k * c)

class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, q):
        # In the forward pass, we use the quantized vectors q
        return q

    @staticmethod
    def backward(ctx, grad_output):
        # In the backward pass, we pass the gradients straight through to z
        return grad_output, None

def hard_discretization(audio_embeds, codebook):
    # audio_embeds: (B, T, D), codebook: (V, D)
    audio_embeds_norm = F.normalize(audio_embeds, p=2, dim=-1)
    codebook_norm = F.normalize(codebook, p=2, dim=-1)
    similarities = torch.einsum('btd,vd->btv', audio_embeds_norm, codebook_norm)
    indices = similarities.argmax(dim=-1)  # (B, T)
    quantized_hard = codebook[indices]  # (B, T, D)
    
    # Apply Straight-Through Estimator
    quantized_ste = StraightThroughEstimator.apply(audio_embeds, quantized_hard)
    return quantized_ste, indices

def soft_discretization(audio_embeds, codebook, k=10):
    # audio_embeds: (B, T, D), codebook: (V, D)
    audio_embeds_norm = F.normalize(audio_embeds, p=2, dim=-1)
    codebook_norm = F.normalize(codebook, p=2, dim=-1)
    similarities = torch.einsum('btd,vd->btv', audio_embeds_norm, codebook_norm)
    topk_sim, topk_indices = similarities.topk(k, dim=-1)  # (B, T, k)
    weights = F.softmax(topk_sim, dim=-1)  # (B, T, k)
    
    # Gather the top-k codebook vectors
    # topk_indices: (B, T, k) -> codebook[topk_indices]: (B, T, k, D)
    topk_codebook_vectors = codebook[topk_indices]
    
    # Weighted sum of the top-k vectors
    # weights: (B, T, k), topk_codebook_vectors: (B, T, k, D)
    # einsum: btk,btkd->btd
    quantized = torch.einsum('btk,btkd->btd', weights, topk_codebook_vectors)
    return quantized, topk_indices


# ----------------------------
# 2.  Custom Dataset
# ----------------------------
class AudioTextDataset(Dataset):
    def __init__(self, data, processor, tokenizer, zip_path=None):
        self.data = data
        self.processor = processor
        self.tokenizer = tokenizer
        self.zip_file = zipfile.ZipFile(zip_path) if zip_path and os.path.exists(zip_path) else None
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = item["audio_filepath"]
        text = item["text"]

        try:
            if self.zip_file:
                with self.zip_file.open(audio_path) as audio_file:
                    waveform, sr = torchaudio.load(io.BytesIO(audio_file.read()))
            else:
                waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None

        if sr != self.processor.feature_extractor.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.processor.feature_extractor.sampling_rate)
        
        # --- Ensure mono ---
        if waveform.shape[0] > 1:  # e.g., stereo -> mono
            waveform = waveform.mean(dim=0, keepdim=True)

        # --- Correctly truncate waveform to 30 seconds for Whisper ---
        max_audio_len = 16000 * 30 
        if waveform.shape[1] > max_audio_len:
            waveform = waveform[:, :max_audio_len]

        # Always produce Whisper mel-spectrograms (80, T)
        input_values = self.processor(
            waveform.squeeze(0),
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features[0]

        # Ensure max length of 3000 frames for stability
        max_len = 3000
        if input_values.shape[1] > max_len:
            input_values = input_values[:, :max_len]

        target_ids = self.tokenizer(text, return_tensors="pt", max_length=256, truncation=True).input_ids.squeeze(0)
        
        # Return 80 x T for pad_sequence
        return {"input_values": input_values, "target_ids": target_ids}

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    input_values = pad_sequence([item['input_values'].T for item in batch], batch_first=True, padding_value=0.0).transpose(1, 2)
    target_ids = pad_sequence([item['target_ids'] for item in batch], batch_first=True, padding_value=-100)

    return {"input_values": input_values, "target_ids": target_ids}


# ----------------------------
# 3.  Wrapper Model
# ----------------------------
class SLAMASR(nn.Module):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def forward(self, inputs_embeds: torch.Tensor, labels: torch.Tensor):
        return self.llm(inputs_embeds=inputs_embeds, labels=labels)

# ----------------------------
# 4.  Evaluation loop
# ----------------------------
@torch.no_grad()
def evaluate(model, projector, speech_encoder, loader, tokenizer, device, dtype, config, discretization_fn, prompt_embeds):
    model.eval()
    projector.eval()
    speech_encoder.eval()
    
    total_loss = 0
    total_wer = 0
    total_samples = 0
    
    pbar = tqdm(loader, desc="Evaluating")
    
    example_outputs = []
    visualization_data = [] # For rich examples

    for batch in pbar:
        if batch is None: continue
        
        input_values = batch['input_values'].to(device)
        target_ids = batch['target_ids'].to(device)
        
        with autocast(dtype=dtype, enabled=torch.cuda.is_available()):
            # 1. Get audio embeddings -> downsample -> project
            audio_embeds_raw = speech_encoder(input_values).last_hidden_state
            audio_embeds_ds = downsample_and_concat(audio_embeds_raw, config.downsample_k)
            projected_embeds = projector(audio_embeds_ds.to(torch.float32))

            # 2. Discretize
            codebook = model.llm.get_input_embeddings().weight
            quantized_embeds, indices = discretization_fn(projected_embeds, codebook)
            
            # 3. Prepare input for LLM using the prompt
            target_embeds = model.llm.get_input_embeddings()(target_ids.clamp(min=0)) # clamp to avoid -100 index
            
            batch_inputs_embeds = []
            batch_labels = []
            for i in range(quantized_embeds.size(0)):
                # [USER_prompt, audio, ASSISTANT_prompt, target]
                inputs_embeds = torch.cat([
                    prompt_embeds['user'], 
                    quantized_embeds[i], 
                    prompt_embeds['assistant'],
                    target_embeds[i]
                ], dim=0)
                batch_inputs_embeds.append(inputs_embeds)

                # Labels: ignore everything except the target
                labels = torch.cat([
                    torch.full(prompt_embeds['user'].shape[:1], -100, device=device),
                    torch.full(quantized_embeds[i].shape[:1], -100, device=device),
                    torch.full(prompt_embeds['assistant'].shape[:1], -100, device=device),
                    target_ids[i]
                ], dim=0)
                batch_labels.append(labels)

            inputs_embeds = pad_sequence(batch_inputs_embeds, batch_first=True)
            labels = pad_sequence(batch_labels, batch_first=True, padding_value=-100)

            # 4. Get loss
            outputs = model(inputs_embeds=inputs_embeds.to(dtype), labels=labels)
            total_loss += outputs.loss.item()

        # Handle different shapes of indices from hard/soft discretization for visualization
        if indices.ndim == 3: # soft-disc returns (B, T, k)
            indices_for_vis = indices[:, :, 0]
        else: # hard-disc returns (B, T)
            indices_for_vis = indices

        # --- Generation for WER and examples ---
        # Prepare prompt for generation: [USER_prompt, audio, ASSISTANT_prompt]
        generation_prompt_embeds = torch.cat([
            prompt_embeds['user'].expand(quantized_embeds.size(0), -1, -1),
            quantized_embeds,
            prompt_embeds['assistant'].expand(quantized_embeds.size(0), -1, -1)
        ], dim=1)

        generated_ids = model.llm.generate(
            inputs_embeds=generation_prompt_embeds.to(dtype),
            max_new_tokens=config.max_new_tokens,
            num_beams=config.beam_size,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        for i in range(len(generated_ids)):
            # Decode only the newly generated tokens
            generated_text = tokenizer.decode(generated_ids[i][generation_prompt_embeds.size(1):], skip_special_tokens=True)
            
            # Filter out padding tokens for target text decoding
            valid_target_ids = [token_id for token_id in target_ids[i] if token_id != -100]
            target_text = tokenizer.decode(valid_target_ids, skip_special_tokens=True)

            if target_text:
                wer = jiwer.wer(target_text, generated_text)
                total_wer += wer
                total_samples += 1

            # Store simple text examples
            if len(example_outputs) < 5:
                 example_outputs.append(f"\n  TARGET    : {target_text}\n  GENERATED : {generated_text}\n")
            
            # Store rich visualization examples
            if len(visualization_data) < 3:
                embed_sample = str(projected_embeds[i, :2, :5].to(torch.float32).cpu().numpy().round(2))
                audio_as_tokens = tokenizer.decode(indices_for_vis[i], skip_special_tokens=True)
                
                visualization_data.append({
                    "target": target_text,
                    "generated": generated_text,
                    "embed_sample": embed_sample,
                    "audio_as_tokens": audio_as_tokens
                })


    pbar.write("\n--- Validation Examples ---")
    for example in random.sample(example_outputs, min(len(example_outputs), 3)):
        pbar.write(example)
    pbar.write("-------------------------\n")
    
    model.train()
    projector.train()
    speech_encoder.train()
    
    avg_loss = total_loss / len(loader)
    avg_wer = (total_wer / total_samples) * 100 if total_samples > 0 else 0
    
    return avg_loss, avg_wer, visualization_data

# ----------------------------
# 5.  Training loop
# ----------------------------
def train(config: TrainingConfig, stage: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    os.makedirs(config.output_dir, exist_ok=True)

    # --- 1. Models & Tokenizer ---
    processor = AutoProcessor.from_pretrained(config.audio_encoder_model)
    speech_encoder = WhisperModel.from_pretrained(config.audio_encoder_model, torch_dtype=dtype).to(device).encoder
    
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    llm = AutoModelForCausalLM.from_pretrained(config.llm_model, torch_dtype=dtype, device_map={"": device})

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

    # --- 3. Projector & Prompt ---
    projector = Projector(
        input_dim=speech_encoder.config.hidden_size * config.downsample_k,
        hidden_dim=config.projector_hidden_dim,
        output_dim=llm.config.hidden_size
    ).to(device)

    # Pre-embed the fixed text parts of the prompt
    user_prompt_tokens = tokenizer("USER: ", return_tensors="pt").input_ids.to(device)
    assistant_prompt_tokens = tokenizer(" Transcribe speech to text. ASSISTANT: ", return_tensors="pt").input_ids.to(device)
    prompt_embeds = {
        'user': llm.get_input_embeddings()(user_prompt_tokens).squeeze(0),
        'assistant': llm.get_input_embeddings()(assistant_prompt_tokens).squeeze(0)
    }

    # --- 4. Stage-specific settings ---
    if stage == 1:
        epochs = config.epochs_stage1
        lr = config.lr_stage1
        batch_size = config.batch_size_stage1
        discretization_fn = hard_discretization
        llm.get_input_embeddings().weight.requires_grad_(False)
        speech_encoder.requires_grad_(False)
        params_to_train = list(projector.parameters()) + [p for p in llm.parameters() if p.requires_grad]

    elif stage == 2:
        epochs = config.epochs_stage2
        lr = config.lr_stage2
        batch_size = config.batch_size_stage2
        discretization_fn = lambda audio_embeds, cb: soft_discretization(audio_embeds, cb, k=config.top_k)
        llm.get_input_embeddings().weight.requires_grad_(True)
        speech_encoder.requires_grad_(False)
        params_to_train = list(projector.parameters()) + [p for p in llm.parameters() if p.requires_grad]
    else:
        raise ValueError(f"Invalid stage: {stage}")

    # --- 5. Dataset & Dataloader ---
    try:
        with open(config.jsonl_path, 'r') as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: JSONL file not found at {config.jsonl_path}")
        return
    
    train_data, val_data = train_test_split(data, test_size=config.val_test_size, random_state=42)
    print(f"Data split: {len(train_data)} training, {len(val_data)} validation.")

    train_dataset = AudioTextDataset(train_data, processor, tokenizer, config.zip_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)

    val_dataset = AudioTextDataset(val_data, processor, tokenizer, config.zip_path)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)

    # --- 6. Optimizer, Scheduler, Scaler ---
    optimizer = torch.optim.AdamW(params_to_train, lr=lr)

    # Scheduler and Scaler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / config.warmup_steps))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    model = SLAMASR(llm)

    # --- 7. Checkpoint Loading ---
    start_epoch = 0
    start_batch_idx = 0
    global_step = 0
    best_val_loss = float('inf')
    
    checkpoint_path = os.path.join(config.output_dir, f"stage_{stage}_latest.pt")
    if os.path.exists(checkpoint_path) and config.resume:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        projector.load_state_dict(checkpoint['projector_state_dict'])
        adapter_path_latest = os.path.join(config.output_dir, "latest_adapter")
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
        print(f"Resumed from Epoch {start_epoch}, Step {global_step}, Batch {start_batch_idx}")
    elif stage == 2:
        stage1_checkpoint_path = os.path.join(config.output_dir, "stage_1_best.pt")
        if os.path.exists(stage1_checkpoint_path):
            print(f"Initializing Stage 2 with best checkpoint from Stage 1: {stage1_checkpoint_path}")
            checkpoint = torch.load(stage1_checkpoint_path, map_location=device)
            projector.load_state_dict(checkpoint['projector_state_dict'])
            adapter_path_best = os.path.join(config.output_dir, "best_adapter")
            if os.path.isdir(adapter_path_best):
                llm.load_adapter(adapter_path_best, adapter_name="default", is_trainable=True)
                print(f"Loaded adapter from {adapter_path_best}.")
            else:
                print(f"Adapter directory {adapter_path_best} not found. Proceeding without loading adapter.")
        else:
            print("Warning: Stage 1 checkpoint not found for Stage 2 initialization.")

    # --- 8. Training Loop ---
    for epoch in range(start_epoch, epochs):
        model.train()
        projector.train()
        if stage == 1:
            speech_encoder.eval()
        elif stage == 2:
            speech_encoder.eval()
            llm.get_input_embeddings().weight.requires_grad_(True)
        
        # --- Handle Resumption ---
        current_train_data = train_data
        
        if epoch == start_epoch and start_batch_idx > 0 and config.resume:
            samples_to_skip = start_batch_idx * batch_size
            if samples_to_skip < len(train_data):
                print(f"Resuming epoch {epoch}. Skipping {samples_to_skip} samples.")
                current_train_data = train_data[samples_to_skip:]
            else:
                print(f"Epoch {epoch} already completed. Starting next epoch.")
                start_batch_idx = 0  # Reset for next epoch
                continue  # Skip to the next epoch
        
        # Recreate dataset and dataloader for the current epoch state
        epoch_train_dataset = AudioTextDataset(current_train_data, processor, tokenizer, config.zip_path)
        epoch_train_loader = DataLoader(epoch_train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)
        
        initial_batch_idx = start_batch_idx if epoch == start_epoch else 0
        total_batches = len(epoch_train_loader) + initial_batch_idx
        pbar = tqdm(epoch_train_loader, desc=f"Epoch {epoch+1}/{epochs} [Stage {stage}]", initial=initial_batch_idx, total=total_batches)
        
        for i, batch in enumerate(pbar, start=initial_batch_idx):
            is_update_step = (i + 1) % config.gradient_accumulation_steps == 0
            
            if batch is None: continue
            
            input_values = batch['input_values'].to(device) # Shape: (B, 80, T_max)
            target_ids = batch['target_ids'].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=torch.cuda.is_available()):
                # --- Feature Normalization ---
                mean = input_values.mean(dim=-1, keepdim=True)
                std = input_values.std(dim=-1, keepdim=True)
                input_values_normalized = (input_values - mean) / (std + 1e-6)

                # 1. Get audio embeddings -> downsample -> project
                audio_embeds_raw = speech_encoder(input_values_normalized).last_hidden_state
                audio_embeds_ds = downsample_and_concat(audio_embeds_raw, config.downsample_k)
                projected_embeds = projector(audio_embeds_ds.to(torch.float32))

                # 2. Discretize
                codebook = llm.get_input_embeddings().weight
                quantized_embeds, indices = discretization_fn(projected_embeds, codebook)
                
                # --- Cosine Sim Metric ---
                cosine_sim_metric = F.cosine_similarity(projected_embeds, quantized_embeds.detach(), dim=-1).mean().item()

                # 3. Prepare for LLM
                target_embeds = llm.get_input_embeddings()(target_ids.clamp(min=0))
                
                inputs_embeds = torch.cat([
                    prompt_embeds['user'].expand(quantized_embeds.size(0), -1, -1),
                    quantized_embeds,
                    prompt_embeds['assistant'].expand(quantized_embeds.size(0), -1, -1),
                    target_embeds
                ], dim=1)

                labels = torch.cat([
                    torch.full((quantized_embeds.size(0), prompt_embeds['user'].size(0)), -100, device=device),
                    torch.full((quantized_embeds.size(0), quantized_embeds.size(1)), -100, device=device),
                    torch.full((quantized_embeds.size(0), prompt_embeds['assistant'].size(0)), -100, device=device),
                    target_ids
                ], dim=1)

                outputs = model(inputs_embeds=inputs_embeds, labels=labels)
                loss = outputs.loss / config.gradient_accumulation_steps

            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at step {global_step}. Skipping step.")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            
            if is_update_step:
                # --- Grad Norm Metrics ---
                grad_norm_before_clip = 0
                for p in params_to_train:
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        grad_norm_before_clip += param_norm.item() ** 2
                grad_norm_before_clip = (grad_norm_before_clip ** 0.5) if grad_norm_before_clip > 0 else 0.0

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params_to_train, 1.0)
                
                grad_norm_after_clip = 0
                for p in params_to_train:
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        grad_norm_after_clip += param_norm.item() ** 2
                grad_norm_after_clip = (grad_norm_after_clip ** 0.5) if grad_norm_after_clip > 0 else 0.0
                # --- End Metrics ---

                scaler.step(optimizer)
                scaler.update()
            
                if global_step < config.warmup_steps:
                    scheduler.step()
            
                global_step += 1
                
                # Update postfix with all metrics
                gpu_mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
                pbar.set_postfix({
                    "loss": f"{loss.item()*config.gradient_accumulation_steps:.3f}", 
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "gn": f"{grad_norm_after_clip:.2f}",
                    "cos_sim": f"{cosine_sim_metric:.2f}",
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
                            "cosine_sim_proj_quant": cosine_sim_metric,
                            "t_max": input_values.shape[2],
                            "gpu_mem_allocated_gb": gpu_mem_alloc,
                            "gpu_mem_max_allocated_gb": gpu_mem_max,
                            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })

                # --- Checkpointing & Validation ---
                if global_step > 0 and global_step % config.save_every_n_steps == 0:
                    latest_checkpoint_path = os.path.join(config.output_dir, f"stage_{stage}_latest.pt")
                    torch.save({
                        'epoch': epoch, 
                        'batch_idx': i,
                        'global_step': global_step,
                        'projector_state_dict': projector.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'best_val_loss': best_val_loss,
                    }, latest_checkpoint_path)
                    llm.save_pretrained(os.path.join(config.output_dir, "latest_adapter"))
                    pbar.write(f"\n--- Checkpoint Saved ---\nSaved latest model at step {global_step}.\n------------------------\n")

                if global_step > 0 and global_step % config.val_every_n_steps == 0:
                    pbar.write(f"\nRunning validation at step {global_step}...")
                    val_loss, val_wer, vis_data = evaluate(model, projector, speech_encoder, val_loader, tokenizer, device, dtype, config, discretization_fn, prompt_embeds)
                    pbar.write(f"Step {global_step} | Validation Loss: {val_loss:.3f} | Validation WER: {val_wer:.2f}%")
                    
                    pbar.write("\n--- Embedding Visualization Examples ---")
                    for item in vis_data:
                        pbar.write(f"  TARGET         : {item['target']}")
                        pbar.write(f"  GENERATED      : {item['generated']}")
                        pbar.write(f"  EMBEDS (sample): {item['embed_sample']}")
                        pbar.write(f"  AUDIO->TOKENS  : {item['audio_as_tokens'][:200]}...")
                    pbar.write("-------------------------------------\n")

                    if wandb.run and not wandb.run.disabled:
                        wandb.log({"val/loss": val_loss, "val/wer": val_wer, "step": global_step})
                        vis_table = wandb.Table(columns=["Target", "Generated", "Embed Sample", "Audio-as-Tokens"])
                        for item in vis_data:
                            vis_table.add_data(item['target'], item['generated'], item['embed_sample'], item['audio_as_tokens'])
                        wandb.log({"validation/examples": vis_table, "step": global_step})
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_checkpoint_path = os.path.join(config.output_dir, f"stage_{stage}_best.pt")
                        torch.save({
                            'epoch': epoch,
                            'global_step': global_step,
                            'projector_state_dict': projector.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                            'best_val_loss': best_val_loss,
                        }, best_checkpoint_path)
                        llm.save_pretrained(os.path.join(config.output_dir, "best_adapter"))
                        pbar.write(f"--- New Best Model ---\nSaved new best model at step {global_step} with val_loss: {best_val_loss:.3f}\n----------------------\n")
        
        # Reset start_batch_idx for next epoch
        start_batch_idx = 0

# ----------------------------
# 6.  Entry-point
# ----------------------------
if __name__ == "__main__":
    # --- CHOOSE STAGE ---
    STAGE = 1 # or 2
    # --------------------

    config = TrainingConfig()
    
    # Override config for different stages if needed
    if STAGE == 1:
        # Effective batch size of 32
        config.gradient_accumulation_steps = 32 // config.batch_size_stage1
    elif STAGE == 2:
        # Effective batch size of 16
        config.gradient_accumulation_steps = 16 // config.batch_size_stage2
        config.wandb_name = "stage2-run"
        config.resume = True

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
            wandb.define_metric("val/wer", step_metric="step")
            wandb.define_metric("learning_rate", step_metric="step")
        except Exception as e:
            print(f"WandB login or init failed: {e}. Running in disabled mode.")
            wandb.init(mode="disabled")
    else:
        print("WANDB_API_KEY not found. Running wandb in disabled mode.")
        wandb.init(mode="disabled")
    
    train(config, stage=STAGE) 