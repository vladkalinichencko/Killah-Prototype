import sys
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLM:
    _instance = None
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        # Read model directory from environment variable
        base_model_path = os.environ.get('MODEL_DIR') or os.path.dirname(__file__)
        self.model_dir = os.path.join(base_model_path, "gemma-3-4b-pt-q8bits")
    
    def load_model(self):
        """Loads the model and tokenizer, returning True if successful, False otherwise."""
        if self.model and self.tokenizer:
            return True
        try:
            print(f"Loading tokenizer and model from: {self.model_dir}", file=sys.stderr, flush=True)
            hf_id = "poinka/gemma-3-4b-pt-q8bits"
            required_files = ["tokenizer.model", "model.safetensors"]

            if all(os.path.exists(os.path.join(self.model_dir, f)) for f in required_files):
                print(f"Loading local model from {self.model_dir}", file=sys.stderr, flush=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_dir,
                    torch_dtype=torch.bfloat16,
                    device_map=device
                )
            else:
                print(f"Local model incomplete or missing at {self.model_dir}. Downloading from Hugging Face...", file=sys.stderr, flush=True)
                self.tokenizer = AutoTokenizer.from_pretrained(hf_id, cache_dir=self.model_dir)
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
                self.model = AutoModelForCausalLM.from_pretrained(
                    hf_id,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                    cache_dir=self.model_dir
                )
            print("Model and tokenizer loaded successfully", file=sys.stderr, flush=True)
            return True
        except Exception as e:
            print(f"ERROR loading model or tokenizer: {e}", file=sys.stderr, flush=True)
            return False

    def get_model(self):
        """Returns the loaded model."""
        return self.model

    def get_tokenizer(self):
        """Returns the loaded tokenizer."""
        return self.tokenizer

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
