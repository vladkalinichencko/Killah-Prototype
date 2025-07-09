import os
import json
import zipfile
import io
from datetime import datetime
import math

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
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
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
    llm_model: str = "google/gemma-3-4b-pt"
    audio_encoder_model: str = "openai/whisper-small"
    output_dir: str = "checkpoints_audio_llm"
    jsonl_path: str = "transcripts.jsonl"
    zip_path: str = "LibriSpeech.zip"

    # Q-Former Projector
    qformer_num_query_tokens: int = 64
    qformer_num_heads: int = 8
    qformer_num_layers: int = 4
    qformer_hidden_dim: int = 2048

    # Loss weights
    contrastive_loss_weight: float = 1.0
    matching_loss_weight: float = 1.0

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"])
    init_lora_weights: bool = True

    # Discretization -> REMOVED
    downsample_k: int = 5

    # Training parameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    epochs: int = 5
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    val_test_size: float = 0.05
    
    # Checkpointing & Logging
    resume: bool = True
    val_every_n_steps: int = 500
    save_every_n_steps: int = 50
    log_every_n_steps: int = 5
    
    # Generation
    max_new_tokens: int = 70
    beam_size: int = 4

    # W&B
    wandb_project: str = "audio-llm-qformer"
    wandb_name: str = "qformer-run"
    wandb_api_key: str = ""

    # Quantization
    quantize_4bit: bool = True
    bnb_compute_dtype: str = "bfloat16"
    hf_token: str = ""

# ----------------------------
# 1.  Audio -> LLM Bridge
# ----------------------------

class QFormerProjector(nn.Module):
    """
    A Q-Former based projector.
    It projects a sequence of audio embeddings to a fixed number of output embeddings.
    It uses a number of learnable query embeddings that attend to the audio features.
    """
    def __init__(self, num_query_tokens: int, input_dim: int, output_dim: int, num_heads: int, num_layers: int, hidden_dim: int):
        super().__init__()
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, output_dim))

        # Project input audio features to the Q-Former's dimension
        self.input_proj = nn.Linear(input_dim, output_dim)

        # Transformer decoder layers that will perform cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation=F.gelu,
            batch_first=True,
            norm_first=True  # Pre-LayerNorm for stability
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Q-Former.
        Args:
            x (torch.Tensor): Input audio features of shape (B, T, D_input).
        Returns:
            torch.Tensor: Output embeddings of shape (B, N_query, D_output).
        """
        # Project audio features to the same dimension as the query tokens
        memory = self.input_proj(x.to(self.input_proj.weight.dtype))

        # Expand query tokens for the batch
        query_tokens = self.query_tokens.expand(x.shape[0], -1, -1)

        # The queries attend to the audio features (memory)
        output = self.transformer_decoder(tgt=query_tokens, memory=memory)
        return output

def downsample_and_concat(x: torch.Tensor, k: int) -> torch.Tensor:
    """Concatenate every k consecutive frames."""
    b, t, c = x.shape
    if t == 0:
        return x.view(b, 0, c)
    n = t // k
    x = x[:, :n * k, :].reshape(b, n, k, c)
    return x.reshape(b, n, k * c)

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
    def __init__(self, llm, atm_head_hidden_dim: int):
        super().__init__()
        self.llm = llm
        # Audio-Text Matching Head
        llm_hidden_size = llm.config.hidden_size
        self.atm_head = nn.Sequential(
            nn.Linear(llm_hidden_size, atm_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(atm_head_hidden_dim, 1)
        )

    def forward(self, inputs_embeds: torch.Tensor, labels: torch.Tensor):
        return self.llm(inputs_embeds=inputs_embeds, labels=labels)

# ----------------------------
# 4.  Evaluation loop
# ----------------------------
@torch.no_grad()
def evaluate(model, projector, speech_encoder, loader, tokenizer, device, dtype, config, prompt_embeds):
    model.eval()
    projector.eval()
    speech_encoder.eval()
    
    total_loss = 0
    total_wer = 0
    total_samples = 0
    
    pbar = tqdm(loader, desc="Evaluating")
    
    example_outputs = []

    for batch in pbar:
        if batch is None: continue
        
        input_values = batch['input_values'].to(device)
        target_ids = batch['target_ids'].to(device)
        
        with autocast(dtype=dtype, enabled=torch.cuda.is_available()):
            # 1. Get audio embeddings -> downsample -> project with Q-Former
            audio_embeds_raw = speech_encoder(input_values).last_hidden_state
            audio_embeds_ds = downsample_and_concat(audio_embeds_raw, config.downsample_k)
            projected_embeds = projector(audio_embeds_ds.to(torch.float32))
            
            # 2. Prepare input for LLM using the prompt
            target_embeds = model.llm.get_input_embeddings()(target_ids.clamp(min=0)) # clamp to avoid -100 index
            
            batch_inputs_embeds = []
            batch_labels = []
            for i in range(projected_embeds.size(0)):
                # [USER_prompt, audio, ASSISTANT_prompt, target]
                inputs_embeds = torch.cat([
                    prompt_embeds['user'], 
                    projected_embeds[i], 
                    prompt_embeds['assistant'],
                    target_embeds[i]
                ], dim=0)
                batch_inputs_embeds.append(inputs_embeds)

                # Labels: ignore everything except the target
                labels = torch.cat([
                    torch.full(prompt_embeds['user'].shape[:1], -100, device=device),
                    torch.full(projected_embeds[i].shape[:1], -100, device=device),
                    torch.full(prompt_embeds['assistant'].shape[:1], -100, device=device),
                    target_ids[i]
                ], dim=0)
                batch_labels.append(labels)

            inputs_embeds = pad_sequence(batch_inputs_embeds, batch_first=True)
            labels = pad_sequence(batch_labels, batch_first=True, padding_value=-100)

            # 3. Get loss
            outputs = model(inputs_embeds=inputs_embeds.to(dtype), labels=labels)
            total_loss += outputs.loss.item()

        # --- Generation for WER and examples ---
        # Prepare prompt for generation: [USER_prompt, audio, ASSISTANT_prompt]
        generation_prompt_embeds = torch.cat([
            prompt_embeds['user'].expand(projected_embeds.size(0), -1, -1),
            projected_embeds,
            prompt_embeds['assistant'].expand(projected_embeds.size(0), -1, -1)
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

    pbar.write("\n--- Validation Examples ---")
    for example in random.sample(example_outputs, min(len(example_outputs), 3)):
        pbar.write(example)
    pbar.write("-------------------------\n")
    
    model.train()
    projector.train()
    speech_encoder.train()
    
    avg_loss = total_loss / len(loader)
    avg_wer = (total_wer / total_samples) * 100 if total_samples > 0 else 0
    
    return avg_loss, avg_wer

# ----------------------------
# 5.  Training loop
# ----------------------------
def train(config: TrainingConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    os.makedirs(config.output_dir, exist_ok=True)

    # --- 1. Models & Tokenizer ---
    processor = AutoProcessor.from_pretrained(config.audio_encoder_model, token=config.hf_token)
    speech_encoder = WhisperModel.from_pretrained(config.audio_encoder_model, torch_dtype=dtype, token=config.hf_token).to(device).encoder
    
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model, padding_side="left", token=config.hf_token)
    # Ensure pad_token differs from eos_token to avoid immediate generation stop
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
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
            torch_dtype=dtype,
            device_map={"": device},
            attn_implementation="eager",
        )
    # Resize token embeddings to accommodate newly added pad token
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

    # --- 3. Projector & Prompt ---
    # Determine LLM embedding dimension robustly (handles models without `config.hidden_size`)
    try:
        llm_hidden = llm.config.hidden_size  # works for most models
    except AttributeError:
        # Fallback to embedding matrix shape
        llm_hidden = llm.get_input_embeddings().weight.shape[1]

    projector = QFormerProjector(
        num_query_tokens=config.qformer_num_query_tokens,
        input_dim=speech_encoder.config.hidden_size * config.downsample_k,
        output_dim=llm_hidden,
        num_heads=config.qformer_num_heads,
        num_layers=config.qformer_num_layers,
        hidden_dim=config.qformer_hidden_dim
    ).to(device)

    # Pre-embed the fixed text parts of the prompt
    user_prompt_tokens = tokenizer("USER: ", return_tensors="pt").input_ids.to(device)
    assistant_prompt_tokens = tokenizer(" Transcribe speech to text. ASSISTANT: ", return_tensors="pt").input_ids.to(device)
    prompt_embeds = {
        'user': llm.get_input_embeddings()(user_prompt_tokens).squeeze(0),
        'assistant': llm.get_input_embeddings()(assistant_prompt_tokens).squeeze(0)
    }

    # --- 4. Training setup ---
    llm.get_input_embeddings().weight.requires_grad_(True)
    speech_encoder.requires_grad_(False)
    params_to_train = list(projector.parameters()) + [p for p in llm.parameters() if p.requires_grad]

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
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)

    val_dataset = AudioTextDataset(val_data, processor, tokenizer, config.zip_path)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)

    # --- 6. Optimizer, Scheduler, Scaler ---
    optimizer = torch.optim.AdamW(params_to_train, lr=config.learning_rate)

    # Scheduler and Scaler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / config.warmup_steps))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    model = SLAMASR(llm, atm_head_hidden_dim=256).to(device)

    # Add ATM head parameters to the optimizer
    params_to_train.extend(model.atm_head.parameters())
    optimizer = torch.optim.AdamW(params_to_train, lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / config.warmup_steps))

    # --- 7. Checkpoint Loading ---
    start_epoch = 0
    start_batch_idx = 0
    global_step = 0
    best_val_loss = float('inf')
    
    checkpoint_path = os.path.join(config.output_dir, "latest.pt")
    if os.path.exists(checkpoint_path) and config.resume:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        projector.load_state_dict(checkpoint['projector_state_dict'])
        # Load ATM head if it exists in the checkpoint
        if 'atm_head_state_dict' in checkpoint:
            model.atm_head.load_state_dict(checkpoint['atm_head_state_dict'])
            print("Loaded ATM head from checkpoint.")
        print(f"Loaded projector from checkpoint.")
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

    # --- 8. Training Loop ---
    for epoch in range(start_epoch, config.epochs):
        model.train()
        projector.train()
        speech_encoder.eval()
        llm.get_input_embeds().weight.requires_grad_(True)
        
        # --- Handle Resumption ---
        current_train_data = train_data
        
        if epoch == start_epoch and start_batch_idx > 0 and config.resume:
            samples_to_skip = start_batch_idx * config.batch_size
            if samples_to_skip < len(train_data):
                print(f"Resuming epoch {epoch}. Skipping {samples_to_skip} samples.")
                current_train_data = train_data[samples_to_skip:]
            else:
                print(f"Epoch {epoch} already completed. Starting next epoch.")
                start_batch_idx = 0  # Reset for next epoch
                continue  # Skip to the next epoch
        
        # Recreate dataset and dataloader for the current epoch state
        epoch_train_dataset = AudioTextDataset(current_train_data, processor, tokenizer, config.zip_path)
        epoch_train_loader = DataLoader(epoch_train_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)
        
        initial_batch_idx = start_batch_idx if epoch == start_epoch else 0
        total_batches = len(epoch_train_loader) + initial_batch_idx
        pbar = tqdm(epoch_train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", initial=initial_batch_idx, total=total_batches)
        
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

                # --- Multi-modal Loss Calculation ---
                loss_gen = 0
                loss_contrastive = 0
                loss_matching = 0
                
                # We need text embeddings for contrastive and matching losses
                text_input_ids = target_ids.clone()
                text_input_ids[text_input_ids == -100] = tokenizer.pad_token_id
                text_embeds = llm.get_input_embeddings()(text_input_ids)
                
                # ---- 2a. Audio-Text Contrastive (ATC) Loss ----
                # Pool the outputs of Q-Former and text embeddings
                audio_feats = F.normalize(projected_embeds.mean(dim=1), p=2, dim=-1)
                text_feats = F.normalize(text_embeds.mean(dim=1), p=2, dim=-1)
                
                sim_matrix = torch.matmul(audio_feats, text_feats.t()) * llm.logit_scale.exp()
                ground_truth = torch.arange(audio_feats.shape[0], device=device)
                loss_contrastive = (F.cross_entropy(sim_matrix, ground_truth) + F.cross_entropy(sim_matrix.t(), ground_truth)) / 2

                # ---- 2b. Audio-Text Matching (ATM) Loss ----
                with torch.no_grad():
                    # Create negative samples by rolling the text embeddings
                    rolled_text_embeds = torch.roll(text_embeds, shifts=1, dims=0)
                    
                # Positive pairs (audio + correct text)
                positive_output = model.atm_head(projected_embeds.mean(dim=1) + text_feats)
                # Negative pairs (audio + incorrect text)
                negative_output = model.atm_head(projected_embeds.mean(dim=1) + F.normalize(rolled_text_embeds.mean(dim=1), p=2, dim=-1))
                
                atm_logits = torch.cat([positive_output, negative_output], dim=0)
                atm_labels = torch.cat([torch.ones_like(positive_output), torch.zeros_like(negative_output)], dim=0)
                loss_matching = F.binary_cross_entropy_with_logits(atm_logits, atm_labels)
                
                # ---- 2c. Generation Loss ----
                target_embeds = text_embeds
                
                inputs_embeds = torch.cat([
                    prompt_embeds['user'].expand(projected_embeds.size(0), -1, -1),
                    projected_embeds,
                    prompt_embeds['assistant'].expand(projected_embeds.size(0), -1, -1),
                    target_embeds
                ], dim=1)

                labels = torch.cat([
                    torch.full((projected_embeds.size(0), prompt_embeds['user'].size(0)), -100, device=device),
                    torch.full((projected_embeds.size(0), projected_embeds.size(1)), -100, device=device),
                    torch.full((projected_embeds.size(0), prompt_embeds['assistant'].size(0)), -100, device=device),
                    target_ids
                ], dim=1)

                outputs = model(inputs_embeds=inputs_embeds, labels=labels)
                loss_gen = outputs.loss
                
                # Total Loss
                loss = (loss_gen + 
                        config.contrastive_loss_weight * loss_contrastive + 
                        config.matching_loss_weight * loss_matching
                       ) / config.gradient_accumulation_steps

            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at step {global_step}. Skipping step.")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            
            if is_update_step:
                scaler.unscale_(optimizer)
                
                # --- Grad Norm Metrics ---
                grad_norm_before_clip = 0
                for p in params_to_train:
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        grad_norm_before_clip += param_norm.item() ** 2
                grad_norm_before_clip = (grad_norm_before_clip ** 0.5) if grad_norm_before_clip > 0 else 0.0

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
                    "loss_g": f"{loss_gen.item():.3f}",
                    "loss_c": f"{loss_contrastive.item():.3f}",
                    "loss_m": f"{loss_matching.item():.3f}",
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
                            "train/loss_total": loss.item()*config.gradient_accumulation_steps,
                            "train/loss_generation": loss_gen.item(),
                            "train/loss_contrastive": loss_contrastive.item(),
                            "train/loss_matching": loss_matching.item(),
                            "learning_rate": optimizer.param_groups[0]['lr'], 
                            "step": global_step,
                            "grad_norm_before_clip": grad_norm_before_clip,
                            "grad_norm_after_clip": grad_norm_after_clip,
                            "t_max": input_values.shape[2],
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
                        'projector_state_dict': projector.state_dict(),
                        'atm_head_state_dict': model.atm_head.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'best_val_loss': best_val_loss,
                    }, latest_checkpoint_path)
                    llm.save_pretrained(os.path.join(config.output_dir, "latest_adapter"), token=config.hf_token or None)
                    pbar.write(f"\n--- Checkpoint Saved ---\nSaved latest model at step {global_step}.\n------------------------\n")

                if global_step > 0 and global_step % config.val_every_n_steps == 0:
                    pbar.write(f"\nRunning validation at step {global_step}...")
                    val_loss, val_wer = evaluate(model, projector, speech_encoder, val_loader, tokenizer, device, dtype, config, prompt_embeds)
                    pbar.write(f"Step {global_step} | Validation Loss: {val_loss:.3f} | Validation WER: {val_wer:.2f}%")
                    
                    if wandb.run and not wandb.run.disabled:
                        wandb.log({"val/loss": val_loss, "val/wer": val_wer, "step": global_step})
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_checkpoint_path = os.path.join(config.output_dir, "best.pt")
                        torch.save({
                            'epoch': epoch,
                            'global_step': global_step,
                            'projector_state_dict': projector.state_dict(),
                            'atm_head_state_dict': model.atm_head.state_dict(),
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
# 6.  Entry-point
# ----------------------------
if __name__ == "__main__":
    config = TrainingConfig()
    
    config.gradient_accumulation_steps = 32 // config.batch_size

    if config.wandb_api_key:
        try:
            wandb.login(key=config.wandb_api_key)
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_name,
                resume="allow",
                config=vars(config),
            )
            wandb.define_metric("train/loss_total", step_metric="step")
            wandb.define_metric("train/loss_generation", step_metric="step")
            wandb.define_metric("train/loss_contrastive", step_metric="step")
            wandb.define_metric("train/loss_matching", step_metric="step")
            wandb.define_metric("val/loss", step_metric="step")
            wandb.define_metric("val/wer", step_metric="step")
            wandb.define_metric("learning_rate", step_metric="step")
        except Exception as e:
            print(f"WandB login or init failed: {e}. Running in disabled mode.")
            wandb.init(mode="disabled")
    else:
        print("WANDB_API_KEY not found. Running wandb in disabled mode.")
        wandb.init(mode="disabled")
    
    train(config) 