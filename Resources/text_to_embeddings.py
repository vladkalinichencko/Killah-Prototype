import sys
import os
import torch
import select
import json
from main_llm import get_model_loader

# Add script directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

class TextEmbeddingGenerator:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"
        loader = get_model_loader()
        if not loader:
            print("Failed to get model loader.", file=sys.stderr, flush=True)
            return None
        
        self.model = loader.get_model()
        if self.model:
            print("Embedding model initialized successfully.", file=sys.stderr, flush=True)

    def generate_embeddings(self, text):
        """Generate embeddings from text input."""
        if not self.model:
            print("Model not initialized", file=sys.stderr, flush=True)
            return None
        try:
            embeddings = self.model.embed(text)
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
            if embeddings_tensor.dim() > 1:
                embeddings_tensor = embeddings_tensor.mean(dim=0)  # Average across tokens/sentences
            print(f"Generated embeddings shape: {embeddings_tensor.shape}", file=sys.stderr, flush=True)
            return embeddings_tensor.tolist()
        except Exception as e:
            print(f"Error generating embeddings: {e}. Text {text}", file=sys.stderr, flush=True)
            return None

    def process_text(self, text):
        """Process text and save embeddings to a file."""
        embeddings = self.generate_embeddings(text)
        if embeddings is not None:
            return embeddings
        return None

if __name__ == "__main__":
    generator = TextEmbeddingGenerator()
    print("READY", flush=True)

    while True:
        readable, _, _ = select.select([sys.stdin], [], [], 1.0)
        if readable:
            line = sys.stdin.readline().strip()
            if not line:
                print("EOF received, exiting", file=sys.stderr, flush=True)
                break
            text = line
            print(f"Processing text: {text}", file=sys.stderr, flush=True)
            embeddings = generator.process_text(text)
            if embeddings:
                print(json.dumps(embeddings), flush=True)
                print("END", flush=True)
            else:
                print("Failed to process text", file=sys.stderr, flush=True)
