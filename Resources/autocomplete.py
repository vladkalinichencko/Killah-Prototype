import torch
import sys
import traceback
import os
import select
import time
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import List, Optional
from main_llm import get_model_loader

# Get the model loader
loader = get_model_loader()

# Load the model and tokenizer
model = loader.get_model() if loader else None
tokenizer = loader.get_tokenizer() if loader else None

# Функция для генерации автодополнений
def generate_suggestions(prompt_text: str, persona_vector: Optional[List[float]] = None, max_suggestions: int = 1) -> List[str]:
    try:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
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

# Основной цикл обработки
if model and tokenizer:
    print("Entering main processing loop.", file=sys.stderr, flush=True)
    current_prompt = None
    interrupted = False

    while True:
        try:
            readable, _, _ = select.select([sys.stdin], [], [], 0.05)

            if readable:
                new_prompt_line = sys.stdin.readline()
                if not new_prompt_line:
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
                    suggestions = generate_suggestions(prompt_to_process)
                    for suggestion in suggestions:
                        if interrupted:
                            break
                        words = suggestion.split()
                        for word in words:
                            if interrupted:
                                break
                            print(word + " ", flush=True)
                            time.sleep(0.07)
                    if not interrupted:
                        print("END_SUGGESTIONS", flush=True)
                else:
                    print("END_SUGGESTIONS", flush=True)
        
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            print(f"FATAL Error in main loop: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            break
            
    print("Exiting.", file=sys.stderr, flush=True)
else:
    print("ERROR: Model or tokenizer failed to load. Exiting.", file=sys.stderr, flush=True)

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
