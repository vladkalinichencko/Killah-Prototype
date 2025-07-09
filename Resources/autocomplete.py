import torch
import sys
import traceback
import os
import select
import time
from typing import List, Optional
from main_llm import get_model_loader
from min_p_sampling import MinPLogitsProcessor
from llama_cpp import Llama


# Add the script's directory to the Python path to allow local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

MAX_SUGGESTION_TOKENS = int(os.environ.get("MAX_SUGGESTION_TOKENS", "10"))

def initialize_model():
    """Initializes and returns the model."""
    try:
        print("Initializing autocomplete models...", file=sys.stderr, flush=True)
        loader = get_model_loader()
        if not loader:
            print("Failed to get model loader.", file=sys.stderr, flush=True)
            return None, None
        
        model = loader.get_model()

        if model:
            print("Autocomplete models initialized successfully.", file=sys.stderr, flush=True)
            return model
        else:
            # The loader itself will print more specific errors.
            return None
    except Exception as e:
        print(f"Error during autocomplete model initialization: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return None

# --- Новый стример токенов ---
def stream_suggestions(model: Llama, prompt_text: str, temperature: float, min_p: float = 0.1):
    """Stream incremental tokens using llama_cpp, mimicking TextIteratorStreamer behavior."""
    # Запускаем генерацию с параметром stream=True
    response = model.create_completion(
        prompt=prompt_text,
        max_tokens=MAX_SUGGESTION_TOKENS,
        temperature=temperature,
        min_p=min_p,  # Раскомментировать, если поддерживается API
        stream=True
    )
    
    # Инициализируем переменные для отслеживания сгенерированного текста
    generated_so_far = ""
    prompt_processed = False
    len_so_far = 0
    
    # Итерируемся по кускам ответа
    for chunk in response:
        # Предполагаем, что каждый chunk содержит новый текст в 'choices[0]['text']'
        new_text = chunk['choices'][0]['text']
        
        # Пропускаем пустые куски
        if not new_text.strip():
            continue

        # Проверяем, обработан ли промпт
        if not prompt_processed:
            if len_so_far > len(prompt_text):
                prompt_processed = True
                delta = new_text[len(prompt_text):]
                if delta.strip():
                    yield delta
                continue
            else:
                # Промпт закончился, начинаем выдавать новый текст
                prompt_processed = True
                generated_so_far = new_text
                yield new_text
        else:
            # Вычисляем дельту (новую часть текста)
            if new_text.startswith(generated_so_far):
                delta = new_text[len(generated_so_far):]
                generated_so_far = new_text
                if delta.strip():
                    yield delta
            else:
                # Если структура изменилась, выдаем весь новый текст
                delta = new_text
                generated_so_far = new_text
                if delta.strip():
                    yield delta

# Основной цикл обработки
if __name__ == "__main__":
    print("Autocomplete.py main loop started.", file=sys.stderr, flush=True)
    
    model = initialize_model()
    if model:
        print("READY", flush=True)  # Вывод на stdout

    current_prompt = None
    interrupted = False
    current_temperature = 0.8  # Initial temperature
    
    while True:
        try:
            if not model:
                print("Autocomplete model not initialized. Retrying in 5 seconds.", file=sys.stderr, flush=True)
                time.sleep(5)
                model = initialize_model()
                continue

            readable, _, _ = select.select([sys.stdin], [], [], 0.05)

            if readable:
                new_prompt_line = sys.stdin.readline()
                if not new_prompt_line: # EOF
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
                    print("Streaming suggestions...", flush=True)
                    for token in stream_suggestions(model, prompt_to_process, current_temperature):
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
