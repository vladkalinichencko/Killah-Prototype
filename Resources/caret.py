import sys
import os
import torch
import select
import json
import requests
import time
from main_llm import get_model_loader

# Add script directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

def wait_for_server(server_url, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{server_url}/health", timeout=2)
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    return False

def initialize_model():
    print("Initializing caret model...", file=sys.stderr, flush=True)
    server_url = "http://localhost:8080"
    if wait_for_server(server_url):
        print("Model server is up.", file=sys.stderr, flush=True)
        loader = get_model_loader()
        if not loader:
            print("Failed to get model loader.", file=sys.stderr, flush=True)
            return None
        model = loader.get_model()
        if model:
            print("Caret model initialized successfully.", file=sys.stderr, flush=True)
            return model
        else:
            print("Failed to initialize model.", file=sys.stderr, flush=True)
            return None
    else:
        print("Model server did not start within timeout.", file=sys.stderr, flush=True)
        return None

def stream_from_embeddings_and_prompt(model, embeddings=None, embeddings_type=None, prompt_text=""):
    base_prompt = "<start_of_turn>user\n"
    if embeddings is not None and embeddings_type:
        token = "[AUDIO]" if embeddings_type == "projected_audio_embeds" else "[TEXT]"
        full_prompt = f"{base_prompt}{token} {prompt_text}<end_of_turn>\n<start_of_turn>model\n"
    else:
        full_prompt = f"{base_prompt}{prompt_text}<end_of_turn>\n<start_of_turn>model\n"

    response = model.create_completion(
        prompt=full_prompt,
        max_tokens=100,
        temperature=0.8,
        min_p=0.1,
        stream=True
    )
    
    buffer = ""
    for line_bytes in response:
        try:
            line = line_bytes.decode('utf-8')
        except UnicodeDecodeError:
            continue
        if line.startswith("data: "):
            buffer += line[6:]
            try:
                data = json.loads(buffer)
                buffer = ""
                if "content" in data and data["content"]:
                    content = data["content"]
                    if content.strip():
                        yield content
                    if data.get("stop", False):
                        break
            except json.JSONDecodeError:
                continue

if __name__ == "__main__":
    model = initialize_model()
    if model:
        print("READY", flush=True)

    while True:
        readable, _, _ = select.select([sys.stdin], [], [], 1.0)
        if readable:
            line = sys.stdin.readline().strip()
            if not line:
                print("EOF received, exiting", file=sys.stderr, flush=True)
                break
            embeddings = None
            embeddings_type = None
            prompt = line
            if "|||" in line:
                parts = line.split("|||", 1)
                embeddings_json = parts[0].strip()
                prompt = parts[1].strip()
                try:
                    data = json.loads(embeddings_json)
                    embeddings_type = data.get("type")
                    embeddings = torch.tensor(data.get("embeddings"), dtype=torch.float32)
                    print(f"Received embeddings of type: {embeddings_type}, shape: {embeddings.shape}", file=sys.stderr, flush=True)
                except json.JSONDecodeError as e:
                    print(f"Error parsing embeddings JSON: {e}", file=sys.stderr, flush=True)
                except Exception as e:
                    print(f"Error processing embeddings: {e}", file=sys.stderr, flush=True)
            
            if ":" in prompt:
                text, user_prompt = prompt.split(":", 1)
                prompt = f"{text.strip()} : {user_prompt.strip()}"
            else:
                prompt = prompt.strip()

            print(f"Processing with prompt: {prompt[-100:]}", file=sys.stderr, flush=True)
            if embeddings is not None:
                print(f"Using embeddings of shape: {embeddings.shape}", file=sys.stderr, flush=True)

            print("STREAM", flush=True)
            for token in stream_from_embeddings_and_prompt(model, embeddings, embeddings_type, prompt):
                print(token, flush=True)
            print("END", flush=True)
