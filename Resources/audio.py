import sys
import os
import time
import select

# Add the script's directory to the Python path to allow local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import librosa
import torch
import torch.nn as nn
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, AutoModel

class AudioProjector(nn.Module):
    def __init__(self, audio_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        self.layer1 = nn.Linear(audio_hidden_size, llm_hidden_size * 2)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(llm_hidden_size * 2, llm_hidden_size)

    def forward(self, audio_embeds: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.gelu(self.layer1(audio_embeds)))

def load_audio_file(file_path, target_sr=16000):
    try:
        audio, sr = librosa.load(file_path, sr=target_sr)
        return audio.astype(np.float32)
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}", file=sys.stderr, flush=True)
        return None

# Глобальные переменные для хранения моделей
audio_extractor = None
audio_encoder = None
projector = None
device = None

def initialize_models():
    global audio_extractor, audio_encoder, projector, device
    try:
        print("Initializing audio models...", file=sys.stderr, flush=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get model base path from environment variable, fall back to bundled resources
        base_model_path = os.environ.get('MODEL_DIR') or os.path.dirname(__file__)
        model_path = os.path.join(base_model_path, "wav2vec2-xls-r-300m")
        
        # Check if critical files exist
        required_files = ["preprocessor_config.json", "pytorch_model.bin"]
        if all(os.path.exists(os.path.join(model_path, f)) for f in required_files):
            audio_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
            audio_encoder = AutoModel.from_pretrained(model_path).to(device)
        else:
            # Пытаемся скачать модель из Hugging Face, если локальная копия отсутствует
            print(f"Model directory {model_path} does not exist. Downloading from Hugging Face…", file=sys.stderr, flush=True)
            hf_id = "facebook/wav2vec2-xls-r-300m"
            audio_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hf_id)
            audio_encoder = AutoModel.from_pretrained(hf_id).to(device)
        projector = AudioProjector(audio_encoder.config.hidden_size, 2560).to(device)
        print("Audio models initialized successfully", file=sys.stderr, flush=True)
        return True
    except Exception as e:
        print(f"Error initializing audio models: {e}", file=sys.stderr, flush=True)
        return False
    
def process_audio_file(file_path):
    global audio_extractor, audio_encoder, projector, device
    if not audio_extractor or not audio_encoder or not projector:
        print("Models not initialized", file=sys.stderr, flush=True)
        return None

    audio_data = load_audio_file(file_path)
    if audio_data is None:
        return None

    try:
        print(f"Processing audio file: {file_path}", file=sys.stderr, flush=True)
        audio_processed = audio_extractor([audio_data], return_tensors="pt", sampling_rate=16000, padding=True)
        audio_values = audio_processed.input_values.to(device)

        with torch.no_grad():
            audio_embeds = audio_encoder(audio_values).last_hidden_state
            projected_audio = projector(audio_embeds)

        output_path = file_path.replace(".wav", "_embeddings.pt")
        torch.save({
            'raw_audio_embeds': audio_embeds,
            'projected_audio_embeds': projected_audio
        }, output_path)

        print(f"Embeddings saved to {output_path}", file=sys.stderr, flush=True)
        print("END_SUGGESTIONS", flush=True)
        return output_path
    except Exception as e:
        print(f"Error processing audio: {e}", file=sys.stderr, flush=True)
        return None

if __name__ == "__main__":
    print("Audio.py main loop started.", file=sys.stderr, flush=True)
    
    models_initialized = initialize_models()
    
    if models_initialized:
        print("READY", flush=True)  # Вывод на stdout
    
    while True:
        try:
            if not models_initialized:
                print("Audio models not initialized. Retrying in 5 seconds.", file=sys.stderr, flush=True)
                time.sleep(5)
                models_initialized = initialize_models()
                continue

            readable, _, _ = select.select([sys.stdin], [], [], 1.0)
            if readable:
                line = sys.stdin.readline()
                if not line: 
                    print("EOF received, exiting audio.py.", file=sys.stderr, flush=True)
                    break 

                file_path = line.strip()
                if not file_path:
                    continue
                
                print(f"Received audio file path: {file_path}", file=sys.stderr, flush=True)
                result = process_audio_file(file_path)
                if result is None:
                    print(f"Processing failed for {file_path}", file=sys.stderr, flush=True)

        except KeyboardInterrupt:
            print("KeyboardInterrupt received, exiting.", file=sys.stderr, flush=True)
            break
        except Exception as e:
            print(f"Fatal error in audio.py main loop: {e}", file=sys.stderr, flush=True)
            time.sleep(5)
