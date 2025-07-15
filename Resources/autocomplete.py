import torch
import sys
import traceback
import os
import select
import time
import requests
import json
from typing import List, Optional
from main_llm import get_model_loader

# Add the script's directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

MAX_SUGGESTION_TOKENS = int(os.environ.get("MAX_SUGGESTION_TOKENS", "10"))

def wait_for_server(server_url, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{server_url}/health", timeout=2)  # Предполагается наличие эндпоинта /health
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    return False
    
def initialize_model():
    print("Initializing autocomplete models...", file=sys.stderr, flush=True)
    server_url = "http://localhost:8080"
    print("Waiting for model server...", file=sys.stderr, flush=True)
    if wait_for_server(server_url):
        print("Model server is up.", file=sys.stderr, flush=True)
        loader = get_model_loader()
        if not loader:
            print("Failed to get model loader.", file=sys.stderr, flush=True)
            return None
        model = loader.get_model()
        if model:
            print("Autocomplete models initialized successfully.", file=sys.stderr, flush=True)
            return model
        else:
            print("Failed to initialize model.", file=sys.stderr, flush=True)
            return None
    else:
        print("Model server did not start within timeout.", file=sys.stderr, flush=True)
        return None

def stream_suggestions(model, prompt_text: str, temperature: float, min_p: float = 0.1):
    response = model.create_completion(
        prompt=prompt_text,
        max_tokens=MAX_SUGGESTION_TOKENS,
        temperature=temperature,
        min_p=min_p,
        stream=True
    )
    buffer = ""  # Буфер для накопления данных
    for line in response:
        if line.startswith("data: "):
            buffer += line[6:]  # Убираем "data: " и добавляем в буфер
            try:
                data = json.loads(buffer)  # Пробуем распарсить JSON
                buffer = ""  # Очищаем буфер после успешного парсинга
                if "content" in data and data["content"].strip():
                    print(f"Yielding token: {data['content']}", file=sys.stderr, flush=True)
                    yield data["content"]  # Передаем токен
                if data.get("stop", False):
                    break  # Останавливаем, если генерация завершена
            except json.JSONDecodeError:
                # Если JSON не полный, продолжаем накапливать
                continue

if __name__ == "__main__":
    print("Autocomplete.py main loop started.", file=sys.stderr, flush=True)
    
    model = initialize_model()
    if model:
        print("READY", flush=True)

    current_prompt = None
    interrupted = False
    current_temperature = 0.8
    
    while True:
        try:
            if not model:
                print("Autocomplete model not initialized. Exiting.", file=sys.stderr, flush=True)
                break

            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if readable:
                new_prompt_line = sys.stdin.readline()
                if not new_prompt_line:
                    print("EOF received, exiting autocomplete.py.", file=sys.stderr, flush=True)
                    break
                
                new_prompt = new_prompt_line.strip()
                if new_prompt.startswith("CMD:"):
                    command = new_prompt[4:]
                    if command == "INCREASE_TEMPERATURE":
                        current_temperature = min(current_temperature + 0.1, 2.0)
                        print(f"Temperature increased to {current_temperature}", file=sys.stderr, flush=True)
                    elif command == "DECREASE_TEMPERATURE":
                        current_temperature = max(current_temperature - 0.1, 0.1)
                        print(f"Temperature decreased to {current_temperature}", file=sys.stderr, flush=True)
                    else:
                        print(f"Unknown command: {command}", file=sys.stderr, flush=True)
                elif new_prompt:
                    current_prompt = new_prompt
                    interrupted = True
                else:
                    current_prompt = None
                    interrupted = True

            if current_prompt:
                prompt_to_process = current_prompt
                current_prompt = None
                interrupted = False

                if prompt_to_process:
                    print("STREAM", flush=True)
                    for token in stream_suggestions(model, prompt_to_process, current_temperature):
                        if interrupted:
                            break
                        print(token, flush=True)
                    if not interrupted:
                        print("END", flush=True)
                else:
                    print("END", flush=True)
        
        except Exception as e:
            print(f"FATAL Error in autocomplete main loop: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            time.sleep(5)
            
    print("Autocomplete.py exited.", file=sys.stderr, flush=True)
