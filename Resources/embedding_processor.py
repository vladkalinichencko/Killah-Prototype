import sys
import os
import torch
import select
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# Add script directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

class EmbeddingProcessor:
    def __init__(self, llm_model, adapter_path, tokenizer_name):
        self.llm_model = llm_model
        self.adapter_path = adapter_path
        self.tokenizer_name = tokenizer_name
        self.device = "mps" if torch.mps.is_available() else "cpu"
        self.llm = None
        self.tokenizer = None
        self.load_models()

    def load_models(self):
        """Load the LLM and tokenizer."""
        try:
            print("Loading LLM and tokenizer...", file=sys.stderr, flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.llm_model,
                torch_dtype=torch.float16,
                device_map="auto"
            ).to(self.device)
            if os.path.exists(self.adapter_path):
                self.llm.load_adapter(self.adapter_path)
                print(f"Loaded adapter from {self.adapter_path}", file=sys.stderr, flush=True)
            else:
                print(f"Adapter path {self.adapter_path} not found", file=sys.stderr, flush=True)
            print("Models loaded successfully", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"Error loading models: {e}", file=sys.stderr, flush=True)
            self.llm = None
            self.tokenizer = None

    def generate_from_embeddings(self, embeddings_file):
        """Generate text from embeddings."""
        if not self.llm or not self.tokenizer:
            print("Models not initialized", file=sys.stderr, flush=True)
            return
        try:
            data = torch.load(embeddings_file, map_location=self.device)
            embeddings = data.get('text_embeds') or data.get('projected_audio_embeds')
            if embeddings is None:
                print("No valid embeddings found in file", file=sys.stderr, flush=True)
                return

            # Prepare prompt embeddings
            user_prompt = "<start_of_turn>user\nGenerate text based on the input:"
            assistant_prompt = "<end_of_turn>\n<start_of_turn>model\n"
            user_tokens = self.tokenizer(user_prompt, return_tensors="pt").input_ids.to(self.device)
            assistant_tokens = self.tokenizer(assistant_prompt, return_tensors="pt").input_ids.to(self.device)
            user_embeds = self.llm.get_input_embeddings()(user_tokens)
            assistant_embeds = self.llm.get_input_embeddings()(assistant_tokens)

            # Ensure embeddings are 3D
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0).unsqueeze(0)
            elif embeddings.dim() == 2:
                embeddings = embeddings.unsqueeze(0)

            input_embeds = torch.cat([user_embeds, embeddings, assistant_embeds], dim=1)

            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
            generation_kwargs = {
                "inputs_embeds": input_embeds,
                "max_new_tokens": 100,
                "do_sample": False,
                "streamer": streamer,
            }
            thread = Thread(target=self.llm.generate, kwargs=generation_kwargs)
            thread.start()
            for token in streamer:
                yield token
        except Exception as e:
            print(f"Error generating text: {шибка}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    llm_model = "google/gemma-3-4b-it"
    adapter_path = os.path.join(script_dir, "checkpoints_audio_llm", "best_adapter")
    tokenizer_name = "google/gemma-3-4b-it"
    
    #processor = EmbeddingProcessor(llm_model, adapter_path, tokenizer_name)
    print("READY", flush=True)

    while True:
        readable, _, _ = select.select([sys.stdin], [], [], 1.0)
        if readable:
            embeddings_file = sys.stdin.readline().strip()
            if not embeddings_file:
                print("EOF received, exiting", file=sys.stderr, flush=True)
                break
            print(f"Processing embeddings from: {embeddings_file}", file=sys.stderr, flush=True)
#            for token in processor.generate_from_embeddings(embeddings_file):
#                print(token, flush=True)
            print("STREAM", flush=True)
            for token in "abc":
                print(token, flush=True)  # Заглушка для проверки 
            print("END", flush=True)
