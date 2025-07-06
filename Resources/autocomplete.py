import torch
import sys
import traceback
import os

# Add the script's directory to the Python path to allow local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import select
import time
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextIteratorStreamer
from typing import List, Optional
from main_llm import get_model_loader
import threading

MAX_SUGGESTION_TOKENS = int(os.environ.get("MAX_SUGGESTION_TOKENS", "100"))

def initialize_models():
    """Initializes and returns the model and tokenizer."""
    try:
        print("Initializing autocomplete models...", file=sys.stderr, flush=True)
        loader = get_model_loader()
        if not loader:
            print("Failed to get model loader.", file=sys.stderr, flush=True)
            return None, None
        
        model = loader.get_model()
        tokenizer = loader.get_tokenizer()

        if model and tokenizer:
            print("Autocomplete models initialized successfully.", file=sys.stderr, flush=True)
            return model, tokenizer
        else:
            # The loader itself will print more specific errors.
            return None, None
    except Exception as e:
        print(f"Error during autocomplete model initialization: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return None, None

# Функция для генерации автодополнений
def generate_suggestions(model, tokenizer, prompt_text: str, persona_vector: Optional[List[float]] = None, max_suggestions: int = 1) -> List[str]:
    try:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_SUGGESTION_TOKENS,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            num_return_sequences=max_suggestions
        )

        suggestions = []
        for output in outputs:
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            suggestion = generated_text[len(prompt_text):].strip().split("\n")[0]
            if suggestion and suggestion not in suggestions:
                suggestions.append(suggestion)
        
        return suggestions[:max_suggestions]
    except Exception as e:
        print(f"Error during generation: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return []

# --- Новый стример токенов ---
def stream_suggestions(model, tokenizer, prompt_text: str):
    """Yield incremental chunks (delta) without repeating the prompt text."""
    try:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

        generation_kwargs = {
            **inputs,
            "max_new_tokens": MAX_SUGGESTION_TOKENS,
            "do_sample": True,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "streamer": streamer,
        }

        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_so_far = ""
        for text in streamer:
            # Remove original prompt prefix if still present
            if prompt_text and text.startswith(prompt_text):
                text = text[len(prompt_text):]

            # If nothing new – skip
            if text == generated_so_far:
                continue

            # Compute delta
            delta = text[len(generated_so_far):] if text.startswith(generated_so_far) else text
            generated_so_far = text

            # Clean leading whitespace only at very first delta
            if not generated_so_far.strip():
                continue
            yield delta

        thread.join()
    except Exception as e:
        print(f"Error during streaming generation: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)

# Основной цикл обработки
if __name__ == "__main__":
    print("Autocomplete.py main loop started.", file=sys.stderr, flush=True)
    
    model, tokenizer = initialize_models()
    if model and tokenizer:
        print("READY", flush=True)  # Вывод на stdout

    current_prompt = None
    interrupted = False

    while True:
        try:
            if not model or not tokenizer:
                print("Autocomplete models not initialized. Retrying in 5 seconds.", file=sys.stderr, flush=True)
                time.sleep(5)
                model, tokenizer = initialize_models()
                continue

            readable, _, _ = select.select([sys.stdin], [], [], 0.05)

            if readable:
                new_prompt_line = sys.stdin.readline()
                if not new_prompt_line: # EOF
                    print("EOF received, exiting autocomplete.py.", file=sys.stderr, flush=True)
                    break
                
                new_prompt = new_prompt_line.strip()
                if new_prompt:
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
                    print("Streaming suggestions...", flush=True)
                    for token in stream_suggestions(model, tokenizer, prompt_to_process):
                        if interrupted:
                            break
                        print(token, flush=True)
                    if not interrupted:
                        print("END_SUGGESTIONS", flush=True)
                else:
                    print("END_SUGGESTIONS", flush=True)
        
        except (EOFError, KeyboardInterrupt):
            print("KeyboardInterrupt or EOF in main loop, exiting.", file=sys.stderr, flush=True)
            break
        except Exception as e:
            print(f"FATAL Error in autocomplete main loop: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            time.sleep(5) # Avoid spamming logs on repeated failure
            
    print("Autocomplete.py exited.", file=sys.stderr, flush=True)

#################################### GGUF Model

# # Путь к локальной папке модели
# here = os.path.abspath(os.path.dirname(__file__))
# model_path = os.path.join(here, "gemma-3-4b-pt-q4_0.gguf")
# # Проверка наличия модели
# if not os.path.exists(model_path):
#     print(f"ERROR: Model not found at {model_path}", file=sys.stderr, flush=True)
#     sys.exit(1)

# # Инициализация модели
# try:
#     # Подавляем все логи загрузки модели
#     with suppress_stderr():
#         llm = Llama(model_path=model_path, n_ctx=512, n_threads=8, verbose=False)
    
#     print("Model loaded successfully", file=sys.stderr, flush=True)
# except Exception as e:
#     print(f"ERROR loading model: {e}", file=sys.stderr, flush=True)
#     sys.exit(1)

# def generate_suggestions(prompt_text: str, persona_vector: Optional[List[float]] = None, max_suggestions: int = 1) -> List[str]:
#     if not prompt_text.strip():
#         return []
#     try:
#         # Логируем промпт на входе
#         print(f"PROMPT: {prompt_text}", file=sys.stderr, flush=True)
        
#         suggestions = []
#         for _ in range(max_suggestions):
#             output = llm(
#                 prompt_text,
#                 max_tokens=15,
#                 temperature=1.0,
#                 top_k=64,  # Рекомендация от Unsloth для Gemma 3
#                 top_p=0.95,
#                 min_p=0.0,
#                 repeat_penalty=1.0,
#                 stop=["<end_of_turn>", "\n"]
#             )
#             suggestion = output["choices"][0]["text"].strip().split("\n")[0]
#             if suggestion and suggestion not in suggestions:
#                 # Логируем каждый выходной токен/предложение
#                 print(f"OUTPUT: {suggestion}", file=sys.stderr, flush=True)
#                 suggestions.append(suggestion)
#         print(f"Generated suggestions: {suggestions}", file=sys.stderr, flush=True)
#         return suggestions
#     except Exception as e:
#         print(f"Error during generation: {e}", file=sys.stderr, flush=True)
#         return []
    
# # Основной цикл обработки
# print("Entering main processing loop.", flush=True)
# current_prompt = None
# interrupted = False


# # --- Фикс: корректная обработка прерывания генерации ---
# while True:
#     try:
#         readable, _, _ = select.select([sys.stdin], [], [], 0.05)

#         if readable:
#             new_prompt_line = sys.stdin.readline()
#             if not new_prompt_line:
#                 break

#             new_prompt = new_prompt_line.strip()
#             print(f"Received prompt: {new_prompt}", file=sys.stderr, flush=True)

#             if new_prompt:
#                 current_prompt = new_prompt
#                 interrupted = True  # <-- выставляем флаг прерывания
#             else:
#                 current_prompt = None
#                 interrupted = True

#         if current_prompt:
#             prompt_to_process = current_prompt
#             current_prompt = None
#             interrupted = False  # <-- Сброс сразу после взятия prompt

#             if prompt_to_process:
#                 print("Streaming suggestions...", flush=True)
#                 suggestions = generate_suggestions(prompt_to_process)
#                 for suggestion in suggestions:
#                     if interrupted:
#                         print("INTERRUPTED", file=sys.stderr, flush=True)
#                         break
#                     words = suggestion.split()
#                     for word in words:
#                         if interrupted:
#                             print("INTERRUPTED", file=sys.stderr, flush=True)
#                             break
#                         print(word + " ", flush=True)
#                         time.sleep(0.07)

#                 if not interrupted:
#                     print("END_SUGGESTIONS", flush=True)
#             else:
#                 print("END_SUGGESTIONS", flush=True)

#     except EOFError:
#         break
#     except KeyboardInterrupt:
#         break
#     except Exception as e:
#         print(f"FATAL Error in main loop: {e}", file=sys.stderr, flush=True)
#         traceback.print_exc(file=sys.stderr)
#         break
