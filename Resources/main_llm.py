import sys
import os
import requests
import json

class ModelProxy:
    def __init__(self, server_url):
        self.server_url = server_url
    
    def embed(self, text):
        try:
            response = requests.post(f"{self.server_url}/embedding", json={"text": text})
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            print(f"Error generating embeddings: {e}", file=sys.stderr, flush=True)
            return None
    
    def create_completion(self, prompt, max_tokens, temperature, min_p, stream=True, lora_adapter=None):
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "min_p": min_p,
            "stream": stream
        }
        if lora_adapter:
            payload["lora_adapters"] = [{"path": lora_adapter, "scale": 1.0}]
        if stream:
            response = requests.post(f"{self.server_url}/completion", json=payload, stream=True)
            response.raise_for_status()
            return response.iter_lines(decode_unicode=False)
        else:
            response = requests.post(f"{self.server_url}/completion", json=payload)
            response.raise_for_status()
            return response.json()

class LLM:
    _instance = None
    
    def __init__(self):
        self.server_url = "http://localhost:8080"
    
    def get_model(self):
        return ModelProxy(self.server_url)

# Singleton instance
_model_loader = None

def get_model_loader():
    global _model_loader
    if _model_loader is None:
        _model_loader = LLM()
    return _model_loader
