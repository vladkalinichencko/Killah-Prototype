import os
import sys
import torch
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "gemma-3-4b-pt-q8bits")
    
    def load_model(self):
        """Loads the model and tokenizer, returning True if successful, False otherwise."""
        # Check if model directory exists
        if not os.path.exists(self.model_dir):
            print(f"ERROR: Model directory not found at {self.model_dir}", file=sys.stderr, flush=True)
            print(f"Current directory: {os.path.dirname(__file__)}", file=sys.stderr, flush=True)
            print(f"Available files: {os.listdir(os.path.dirname(__file__)) if os.path.exists(os.path.dirname(__file__)) else 'Directory not found'}", file=sys.stderr, flush=True)
            return False

        # Initialize model and tokenizer
        try:
            print(f"Loading tokenizer and model from: {self.model_dir}", file=sys.stderr, flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                torch_dtype=torch.bfloat16,
                device_map=device,
                low_cpu_mem_usage=True,
                local_files_only=True
            )
            print("Model and tokenizer loaded successfully", file=sys.stderr, flush=True)
            return True
        except Exception as e:
            print(f"ERROR loading model or tokenizer: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
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
        if _model_loader.load_model():
            print("ModelLoader initialized successfully", file=sys.stderr, flush=True)
        else:
            print("Failed to initialize ModelLoader", file=sys.stderr, flush=True)
    return _model_loader
