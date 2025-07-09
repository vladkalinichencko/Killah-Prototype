import sys
import torch
import os
from llama_cpp import Llama
import contextlib

# Функция для подавления stderr во время загрузки модели
@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


class LLM:
    _instance = None
    
    def __init__(self):
        self.model = None
        # Read model directory from environment variable
        base_model_path = os.environ.get('MODEL_DIR') or os.path.dirname(__file__)
        self.model_path = os.path.join(base_model_path, "gemma-3-4b-pt-q4_0.gguf")
        
        # Получаем HF_TOKEN из переменной окружения
        self.hf_token = os.environ.get('HF_TOKEN')
        if not self.hf_token:
            print("⚠️ HF_TOKEN not set in environment variables. Required for downloading gated model.", file=sys.stderr, flush=True)
    
    def load_model(self):
        """Loads the gguf model using llama_cpp, returning True if successful, False otherwise."""
        if self.model:
            return True
        try:
            print(f"Loading model from: {self.model_path}", file=sys.stderr, flush=True)
            
            # Определяем, использовать ли MPS (Metal)
            use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
            n_gpu_layers = 1 if use_mps else 0  # Используем GPU только если MPS доступен


            hf_id = "google/gemma-3-4b-pt-qat-q4_0-gguf"

            if os.path.exists(self.model_path):
                print(f"Loading local model from {self.model_path}", file=sys.stderr, flush=True)
                with suppress_stderr():
                    self.model = Llama(
                    model_path=self.model_path,
                    n_ctx=1024,
                    n_threads=8,
                    n_gpu_layers = n_gpu_layers,
                    verbose=True)
            else:
                print(f"Local model is missing at {self.model_path}. Downloading from Hugging Face...", file=sys.stderr, flush=True)
                self.model = Llama.from_pretrained(
                    repo_id=hf_id,
                    filename="gemma-3-4b-pt-q4_0.gguf",
                    n_ctx=1024,
                    n_threads=8,
                    n_gpu_layers = n_gpu_layers,
                    verbose=True,
                    token=self.hf_token)
                    
            print("Model loaded successfully", file=sys.stderr, flush=True)
            return True
        except Exception as e:
            print(f"ERROR loading model: {e}", file=sys.stderr, flush=True)
            return False

    def get_model(self):
        """Returns the loaded model."""
        return self.model

# Singleton instance for lazy loading
_model_loader = None

def get_model_loader():
    """Returns the singleton ModelLoader instance, initializing it if necessary."""
    global _model_loader
    if _model_loader is None:
        _model_loader = LLM()
        if not _model_loader.load_model():
            print("Failed to initialize ModelLoader", file=sys.stderr, flush=True)
            _model_loader = None
    return _model_loader
