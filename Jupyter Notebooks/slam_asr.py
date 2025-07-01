import os
import json
import zipfile
import io
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    WhisperProcessor,
    WhisperModel,
)
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
CONFIG = dict(
    # paths & checkpoints
    jsonl_path="transcripts.jsonl",
    zip_path="LibriSpeech.zip", # Optional: if audio files are in a zip
    output_dir="checkpoints",
    resume=True,

    # models
    llm_model="TinyLlama/TinyLlama-1.1B-Chat-v0.4",
    speech_encoder="openai/whisper-small",

    # training hyper-params
    batch_size=28,
    epochs=1,
    lr=1e-4,
    downsample_k=5,
    projector_hidden=2048,

    # validation & checkpointing
    val_every_n_steps=1000,
    save_every_n_steps=50,
    warmup_steps=1000,
    val_test_size=0.05,

    # generation
    max_new_tokens=128,
    max_gen_tokens_eval=70,

    # wandb
    wandb=dict(
        project="slam-asr",
        name="run-3",
        resume="allow",
        entity=None,
        api_key="f2c28a0327b6e9b15d2d3a911be9d6cce58fcd39",
    ),
)

# ----------------------------
# 1.  Speech <-> Text projector
# ----------------------------
class LinearProjector(nn.Module):
    """Two-layer MLP with ReLU used in the SLAM-ASR paper. Runs in float32."""
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
        self.prompt_text = "Transcribe speech to text."

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
        
        input_features = self.processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt").input_features[0]
        
        prompt_ids = self.tokenizer(self.prompt_text, return_tensors="pt").input_ids.squeeze(0)
        target_ids = self.tokenizer(text, return_tensors="pt").input_ids.squeeze(0)
        
        # We will create labels inside the collator
        return {"input_features": input_features, "prompt_ids": prompt_ids, "target_ids": target_ids}

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    
    input_features = pad_sequence([item['input_features'] for item in batch], batch_first=True)
    
    # Pad prompt and target IDs
    prompt_ids = [item['prompt_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]
    
    # No need to pad these here, handled in main loop when creating final labels
    return {"input_features": input_features, "prompt_ids": prompt_ids, "target_ids": target_ids}


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
def evaluate(model, projector, speech_encoder, loader, tokenizer, device, dtype, k, max_new_tokens):
    model.eval()
    projector.eval()
    
    total_loss = 0
    total_wer = 0
    total_samples = 0
    
    pbar = tqdm(loader, desc="Evaluating")
    
    example_outputs = []

    for batch in pbar:
        if batch is None: continue
        
        with autocast(dtype=dtype, enabled=torch.cuda.is_available()):
            input_features = batch['input_features'].to(device)
            audio_embeds_raw = speech_encoder(input_features).last_hidden_state
            audio_embeds_ds = downsample_and_concat(audio_embeds_raw, k)
            projected_embeds = projector(audio_embeds_ds.float())

            batch_inputs_list, batch_labels_list = [], []
            
            for i in range(projected_embeds.size(0)):
                prompt_emb = model.llm.get_input_embeddings()(batch['prompt_ids'][i].to(device))
                target_emb = model.llm.get_input_embeddings()(batch['target_ids'][i].to(device))

                full_emb = torch.cat([projected_embeds[i], prompt_emb, target_emb], dim=0)
                batch_inputs_list.append(full_emb)

                ignore = torch.full((projected_embeds.size(1) + prompt_emb.size(0),), -100, device=device, dtype=torch.long)
                labels = torch.cat([ignore, batch['target_ids'][i].to(device)], dim=0)
                batch_labels_list.append(labels)

            inputs_embeds = pad_sequence(batch_inputs_list, batch_first=True)
            labels = pad_sequence(batch_labels_list, batch_first=True, padding_value=-100)

            outputs = model(inputs_embeds=inputs_embeds, labels=labels)
            total_loss += outputs.loss.item()

        # --- Generation for WER and examples ---
        generated_ids = model.llm.generate(
            inputs_embeds=projected_embeds.to(dtype), # Use projected audio embeds as prompt
            max_new_tokens=max_new_tokens, # Limit generation length
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )

        for i in range(len(generated_ids)):
            # Decode generated and target text
            # The prompt is the audio, so we only decode the new tokens
            generated_text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
            target_text = tokenizer.decode(batch['target_ids'][i], skip_special_tokens=True)

            # Calculate WER
            if target_text:
                wer = jiwer.wer(target_text, generated_text)
                total_wer += wer
                total_samples += 1

            # Store some examples to display later
            if len(example_outputs) < 3:
                 example_outputs.append(f"\n  TARGET    : {target_text}\n  GENERATED : {generated_text}\n")


    # --- Print random examples ---
    print("\n--- Validation Examples ---")
    for example in example_outputs:
        print(example)
    print("-------------------------\n")
    
    model.train()
    projector.train()
    
    avg_loss = total_loss / len(loader)
    avg_wer = (total_wer / total_samples) * 100 if total_samples > 0 else 0
    
    return avg_loss, avg_wer, example_outputs

# ----------------------------
# 5.  Training loop
# ----------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    # --- Create checkpoint directory ---
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # --- 1. Models (LLM and Speech Encoder are frozen) ---
    processor = WhisperProcessor.from_pretrained(CONFIG["speech_encoder"])
    speech_encoder_core = WhisperModel.from_pretrained(CONFIG["speech_encoder"], torch_dtype=dtype).to(device).encoder
    speech_encoder_core.eval()
    for p in speech_encoder_core.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["llm_model"], padding_side="right", pad_token="<|endoftext|>")
    llm = AutoModelForCausalLM.from_pretrained(CONFIG["llm_model"], torch_dtype=dtype, device_map="auto")
    llm.resize_token_embeddings(len(tokenizer))
    llm.requires_grad_(False)
    
    # --- 2. Projector (trainable) ---
    projector = LinearProjector(
        input_dim=speech_encoder_core.config.hidden_size * CONFIG["downsample_k"],
        hidden_dim=CONFIG["projector_hidden"],
        output_dim=llm.config.hidden_size
    ).to(device)

    # --- 3. Dataset & Dataloader ---
    try:
        with open(CONFIG["jsonl_path"], 'r') as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: JSONL file not found at {CONFIG['jsonl_path']}")
        return
    
    # Split data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=CONFIG["val_test_size"], random_state=42)
    print(f"Data split: {len(train_data)} training samples, {len(val_data)} validation samples.")

    train_dataset = AudioTextDataset(train_data, processor, tokenizer, CONFIG["zip_path"])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0)

    val_dataset = AudioTextDataset(val_data, processor, tokenizer, CONFIG["zip_path"])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0)

    # --- 4. Optimizer, scheduler, scaler ---
    optimizer = torch.optim.AdamW(projector.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, step / CONFIG["warmup_steps"]))
    scaler = GradScaler(enabled=torch.cuda.is_available())
    model = SLAMASR(llm)

    # --- 5. Checkpoint Loading ---
    start_epoch = 0
    start_batch_idx = 0
    global_step = 0
    best_val_loss = float('inf')
    
    checkpoint_path = os.path.join(CONFIG["output_dir"], "latest_checkpoint.pt")
    if os.path.exists(checkpoint_path) and CONFIG["resume"]:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        projector.load_state_dict(checkpoint['projector_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_val_loss = checkpoint['best_val_loss']
        # To resume from the next batch, we need to know where we left off
        start_batch_idx = checkpoint['batch_idx'] + 1 
        
        print(f"Resuming from Epoch {start_epoch}, Step {global_step}")

    # --- 6. Training Loop ---
    for epoch in range(start_epoch, CONFIG["epochs"]):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        if start_batch_idx > 0:
            # Update the progress bar description to show we're resuming
            pbar.set_description(f"Epoch {epoch+1}/{CONFIG['epochs']} (resuming from step {global_step})")

        for batch_idx, batch in enumerate(pbar):
            if start_batch_idx > 0:
                if batch_idx < start_batch_idx:
                    continue
                if batch_idx == start_batch_idx:
                    start_batch_idx = 0 # Reset for next epochs
                    pbar.set_description(f"Epoch {epoch+1}/{CONFIG['epochs']}")

            if batch is None: continue
            optimizer.zero_grad(set_to_none=True)

            with autocast(dtype=dtype, enabled=torch.cuda.is_available()):
                input_features = batch['input_features'].to(device)
                
                audio_embeds_raw = speech_encoder_core(input_features).last_hidden_state
                audio_embeds_ds = downsample_and_concat(audio_embeds_raw, CONFIG["downsample_k"])
                
                # Projector runs in float32 for stability
                projected_embeds = projector(audio_embeds_ds.float())

                # Prepare for final cat
                batch_inputs_list = []
                batch_labels_list = []
                
                for i in range(projected_embeds.size(0)):
                    prompt_emb = llm.get_input_embeddings()(batch['prompt_ids'][i].to(device))
                    target_emb = llm.get_input_embeddings()(batch['target_ids'][i].to(device))

                    full_emb = torch.cat([projected_embeds[i], prompt_emb, target_emb], dim=0)
                    batch_inputs_list.append(full_emb)

                    ignore = torch.full((projected_embeds.size(1) + prompt_emb.size(0),), -100, device=device, dtype=torch.long)
                    labels = torch.cat([ignore, batch['target_ids'][i].to(device)], dim=0)
                    batch_labels_list.append(labels)

                # Pad the final combined embeddings and labels
                inputs_embeds = pad_sequence(batch_inputs_list, batch_first=True)
                labels = pad_sequence(batch_labels_list, batch_first=True, padding_value=-100)

                outputs = model(inputs_embeds=inputs_embeds, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})
            print(f"DEBUG W&B: wandb.run is {wandb.run}, disabled: {wandb.run.disabled if wandb.run else 'N/A'}")
            if wandb.run and not wandb.run.disabled:
                wandb.log({"train/loss": loss.item(), "step": global_step}, commit=True)

            # --- Mid-epoch Checkpointing ---
            if global_step > 0 and global_step % CONFIG["save_every_n_steps"] == 0:
                latest_checkpoint_path = os.path.join(CONFIG["output_dir"], "latest_checkpoint.pt")
                torch.save({
                    'epoch': epoch, 
                    'batch_idx': batch_idx,
                    'global_step': global_step,
                    'projector_state_dict': projector.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                }, latest_checkpoint_path)
                print(f"Saved latest checkpoint at step {global_step}")

            # --- Mid-epoch Validation ---
            if global_step > 0 and global_step % CONFIG["val_every_n_steps"] == 0:
                print(f"\nRunning validation at step {global_step}...")
                val_loss, val_wer, val_examples = evaluate(model, projector, speech_encoder_core, val_loader, tokenizer, device, dtype, CONFIG["downsample_k"], CONFIG["max_gen_tokens_eval"])
                print(f"Step {global_step} | Validation Loss: {val_loss:.3f} | Validation WER: {val_wer:.2f}%")

                if wandb.run and not wandb.run.disabled:
                    wandb.log({
                        "val/loss": val_loss,
                        "val/wer": val_wer,
                        "val/examples": "\n".join(val_examples),
                        "step": global_step
                    }, commit=True)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint_path = os.path.join(CONFIG["output_dir"], "best_checkpoint.pt")
                    torch.save({
                        'epoch': epoch + 1,
                        'global_step': global_step,
                        'projector_state_dict': projector.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'best_val_loss': best_val_loss,
                    }, best_checkpoint_path)
                    print(f"New best model saved at step {global_step} with val_loss: {best_val_loss:.3f}")

        
        # --- End of Epoch: Validation & Checkpointing ---
        print("\nRunning end-of-epoch validation...")
        val_loss, val_wer, val_examples = evaluate(model, projector, speech_encoder_core, val_loader, tokenizer, device, dtype, CONFIG["downsample_k"], CONFIG["max_gen_tokens_eval"])
        print(f"Epoch {epoch+1} | Validation Loss: {val_loss:.3f} | Validation WER: {val_wer:.2f}% | Best Val Loss: {best_val_loss:.3f}")

        if wandb.run and not wandb.run.disabled:
            wandb.log({
                "val/loss": val_loss,
                "val/wer": val_wer,
                "val/examples": "\n".join(val_examples),
                "epoch": epoch + 1
            }, commit=True)

        # Save latest checkpoint
        latest_checkpoint_path = os.path.join(CONFIG["output_dir"], "latest_checkpoint.pt")
        torch.save({
            'epoch': epoch + 1, # Save next epoch to start from
            'batch_idx': 0, # Start from the beginning of the next epoch
            'global_step': global_step,
            'projector_state_dict': projector.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss,
        }, latest_checkpoint_path)
        print(f"Saved latest checkpoint at end of epoch {epoch+1}")

        # Save best checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(CONFIG["output_dir"], "best_checkpoint.pt")
            torch.save({
                'epoch': epoch + 1,
                'global_step': global_step,
                'projector_state_dict': projector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss,
            }, best_checkpoint_path)
            print(f"New best model saved with val_loss: {best_val_loss:.3f}")

# ----------------------------
# 6.  Entry-point
# ----------------------------
if __name__ == "__main__":
    api_key = CONFIG["wandb"].get("api_key")
    
    if api_key:
        try:
            wandb.login(key=api_key)
            wandb.init(
                project=CONFIG["wandb"]["project"],
                name=CONFIG["wandb"]["name"],
                entity=CONFIG["wandb"]["entity"],
                resume="allow",
                config=CONFIG,
            )
        except Exception as e:
            print(f"wandb login or init failed: {e}. Running in disabled mode.")
            wandb.init(mode="disabled")
    else:
        print("WANDB_API_KEY not found. Running wandb in disabled mode.")
        wandb.init(mode="disabled")
    
    train() 