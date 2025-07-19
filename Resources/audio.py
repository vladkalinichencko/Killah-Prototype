import sys
import os
import time
import select
import librosa
import torch
import torch.nn as nn
import numpy as np
from transformers import WhisperProcessor, WhisperModel
from huggingface_hub import hf_hub_download

# Add the script's directory to the Python path to allow local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

class AudioProjector(nn.Module):
    """Two-layer MLP with ReLU used in the SLAM-ASR paper."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
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
    
    
# Audio processing class with Whisper Medium and GGUF projector
class AudioProcessor:
    def __init__(self, projector_file: str, downsample_k: int = 4, projector_hidden_dim: int = 2048, llm_hidden_dim: int = 2560):
        self.projector = None
        self.projector_file = projector_file
        self.downsample_k = downsample_k
        self.projector_hidden_dim = projector_hidden_dim
        self.llm_hidden_dim = llm_hidden_dim
        self.whisper_processor = None
        self.whisper_model = None
        self.device = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"
        self.use_mean_pooling = False
        self.load_models()
        
    def load_models(self):
        """Load Whisper Medium and GGUF projector models."""
        try:
            # Load Whisper Medium
            print("Loading Whisper Small model...", file=sys.stderr, flush=True)
            # Get model base path from environment variable, fall back to bundled resources
            base_model_path = os.environ.get('MODEL_DIR') or os.path.dirname(__file__)
            whisper_model_path = os.path.join(base_model_path, "whisper-small")
            # Check if critical files exist
            required_files = ["added_tokens.json", "config.json", "merges.txt", "normalizer.json", "preprocessor_config.json", "pytorch_model.bin", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"]
            
            if all(os.path.exists(os.path.join(whisper_model_path, f)) for f in required_files):
                print(f"Loading local Whisper Small from {whisper_model_path}", file=sys.stderr, flush=True)
                self.whisper_processor = WhisperProcessor.from_pretrained(whisper_model_path)
                self.whisper_model = WhisperModel.from_pretrained(whisper_model_path).to(self.device).encoder
            else:
                # Пытаемся скачать модель из Hugging Face, если локальная копия отсутствует
                print(f"Model directory {whisper_model_path} does not exist. Downloading from Hugging Face…", file=sys.stderr, flush=True)
                hf_id = "openai/whisper-small"
                self.whisper_processor = WhisperProcessor.from_pretrained(hf_id, token=os.environ.get('HF_TOKEN'))
                self.whisper_model = WhisperModel.from_pretrained(hf_id, token=os.environ.get('HF_TOKEN')).to(self.device).encoder
                
            print(f"Whisper loaded successfully", file=sys.stderr, flush=True)
            
            # Load MLP projector
            projector_path = os.path.join(base_model_path, self.projector_file)
            input_dim = self.whisper_model.config.hidden_size * self.downsample_k
            self.projector = AudioProjector(input_dim, self.projector_hidden_dim, self.llm_hidden_dim).to(self.device)
            
            if os.path.exists(projector_path):
                print(f"Loading MLP projector from: {projector_path}", file=sys.stderr, flush=True)
                checkpoint = torch.load(projector_path, map_location=self.device)
                
            else:
                print(f"Model directory {projector_path} does not exist. Downloading from Hugging Face…", file=sys.stderr, flush=True)
                hf_id = "poinka/checkpoints"
                filename = "latest_checkpoint_bs4_epoch_1_step_4300.pt"
                
                print(f"Downloading {filename} from {hf_id}...", file=sys.stderr, flush=True)
                downloaded_path = hf_hub_download(
                    repo_id=hf_id,
                    filename=filename,
                    local_dir=base_model_path,
                )
                print(f"Loading downloaded MLP projector from: {downloaded_path}", file=sys.stderr, flush=True)
                checkpoint = torch.load(downloaded_path, map_location=self0020device)
            
            self.projector.load_state_dict(checkpoint['projector_state_dict'])
            self.projector.eval()
                
            print("Audio models initialized successfully", file=sys.stderr, flush=True)
            return True
        except Exception as e:
            print(f"Error initializing audio models: {e}", file=sys.stderr, flush=True)
            return False
        

    @staticmethod
    def load_audio_file(file_path: str, target_sr: int):
        """Load audio file with librosa."""
        try:
            audio, sr = librosa.load(file_path, sr=target_sr)
            return audio.astype(np.float32)
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}", file=sys.stderr, flush=True)
            return None

    
    def process_audio(self, file_path, target_sr=16000):
        """Process audio file with Whisper Small and generate embeddings using GGUF projector."""
        if not self.whisper_processor or not self.whisper_model or not self.projector:
            print("Models not initialized", file=sys.stderr, flush=True)
            return None

        audio_data = self.load_audio_file(file_path, target_sr)
        if audio_data is None:
            return None

        try:
            print(f"Processing audio file with Whisper Small: {file_path}", file=sys.stderr, flush=True)
            # Generate log-mel spectrograms with WhisperProcessor
            inputs = self.whisper_processor(audio_data, sampling_rate=target_sr, return_tensors="pt").to(self.device)
            
            # Extract encoder embeddings
            with torch.no_grad():
                embeddings = self.whisper_model(**inputs).last_hidden_state # Shape: [1, seq_len, 1024]
                
                # Downsample and concatenate
                downsampled_embeddings = downsample_and_concat(embeddings, self.downsample_k)  # Shape: [1, seq_len//k, 1024*k]
                
                # Project embeddings
                projected_embeddings = self.projector(downsampled_embeddings)  # Shape: [1, seq_len//k, llm_hidden_dim]
                
                # Mean pooling (optional, for compatibility with existing pipeline)
                if self.use_mean_pooling:
                    final_embeddings = torch.mean(projected_embeddings, dim=1).squeeze(0)  # Shape: [llm_hidden_dim]
                else:
                    final_embeddings = projected_embeddings.squeeze(0)  # Shape: [seq_len//k, llm_hidden_dim]
                 
                 
            print(f"Downsampled embeddings shape: {downsampled_embeddings.shape}", file=sys.stderr, flush=True)
            print(f"Projected embeddings shape: {projected_embeddings.shape}", file=sys.stderr, flush=True)
            print(f"Final embeddings shape: {final_embeddings.shape}", file=sys.stderr, flush=True)

            return {"type": "projected_audio_embeds", "embeddings": final_embeddings.tolist()}

        except Exception as e:
            print(f"Error processing audio: {e}", file=sys.stderr, flush=True)
            return None



# Global processor instance
audio_processor = None

def initialize_processor():
    """Initialize the audio processor with Whisper Medium and MLP projector."""
    global audio_processor
    projector_file = "checkpoints/latest_checkpoint_bs4_epoch_1_step_4300.pt"
    audio_processor = AudioProcessor(projector_file)
    
    return audio_processor.projector is not None and audio_processor.whisper_model is not None and audio_processor.whisper_processor is not None

def process_audio_file(file_path):
    """Process the audio file and return embeddings."""
    if not audio_processor or not audio_processor.projector or not audio_processor.whisper_model or not audio_processor.whisper_processor:
        print("Audio processor not initialized", file=sys.stderr, flush=True)
        return None
    result = audio_processor.process_audio(file_path)
    if result is not None:
        print(json.dumps(result), flush=True)
        print("END", flush=True)
    return result
    
if __name__ == "__main__":
    print("Audio.py main loop started.", file=sys.stderr, flush=True)
    
    models_initialized = initialize_processor()
    
    if models_initialized:
        print("READY", flush=True)  # Вывод на stdout
    
    while True:
        try:
            if not models_initialized:
                print("Audio processor not initialized. Retrying in 5 seconds.", file=sys.stderr, flush=True)
                time.sleep(5)
                models_initialized = initialize_processor()
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
