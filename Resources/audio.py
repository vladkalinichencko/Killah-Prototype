import sys
import os
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
        
        # Путь к локальной модели внутри ресурсов приложения
        model_path = os.path.join(os.path.dirname(__file__), "wav2vec2-xls-r-300m")
        if not os.path.exists(model_path):
            print(f"Model directory {model_path} does not exist", file=sys.stderr, flush=True)
            return False

        audio_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        audio_encoder = AutoModel.from_pretrained(model_path).to(device)
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
    print("Entering main processing loop.", flush=True)
    if not initialize_models():  # Проверяем успешность инициализации
        print("Failed to initialize models, exiting.", file=sys.stderr, flush=True)
        sys.exit(1)
    while True:
        try:
            file_path = sys.stdin.readline().strip()
            if not file_path:
                print("Empty input received, exiting.", file=sys.stderr, flush=True)
                break
            print(f"Received audio file path: {file_path}", file=sys.stderr, flush=True)
            result = process_audio_file(file_path)
            if result is None:
                print("Processing failed", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"Fatal error in main loop: {e}", file=sys.stderr, flush=True)
            break
