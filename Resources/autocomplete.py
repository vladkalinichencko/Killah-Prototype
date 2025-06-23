import torch
import sys
import os
import traceback
import select
import time
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional
from llama_cpp import Llama

# Полностью подавляем логи llama_cpp
os.environ["LLAMA_LOG_LEVEL"] = "0"  # Отключаем все логи
os.environ["GGML_LOG_LEVEL"] = "0"   # Отключаем все логи
import contextlib

# Функция для подавления stderr во время загрузки модели
@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr



# Disable Metal Performance Shaders to avoid mach-O errors in app bundle
#os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Force CPU backend to avoid GPU-related issues in packaged app
#torch.backends.mps.is_available = lambda: False
#torch.backends.cuda.is_available = lambda: False

############################## Full Model
# Путь к локальной папке модели
#here = os.path.abspath(os.path.dirname(__file__))
#model_dir = os.path.join(here, "gemma-3-4b-pt")
#
## Проверка наличия модели
#if not os.path.exists(model_dir):
#    print(f"ERROR: Model directory not found at {model_dir}", file=sys.stderr, flush=True)
#    print(f"Current directory: {here}", file=sys.stderr, flush=True)
#    print(f"Available files: {os.listdir(here) if os.path.exists(here) else 'Directory not found'}", file=sys.stderr, flush=True)
#    sys.exit(1)
#
## Инициализация модели и токенизатора
#try:
#    print(f"Loading tokenizer and model from: {model_dir}", file=sys.stderr, flush=True)
#    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
#    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#    model = AutoModelForCausalLM.from_pretrained(
#        model_dir,
#        torch_dtype=torch.bfloat16,
#        device_map=device,
#        low_cpu_mem_usage=True,
#        local_files_only=True
#    )
#    print("Model and tokenizer loaded successfully", file=sys.stderr, flush=True)
#except Exception as e:
#    print(f"ERROR loading model or tokenizer: {e}", file=sys.stderr, flush=True)
#    traceback.print_exc(file=sys.stderr)
#    sys.exit(1)
#
## Функция для генерации автодополнений
#def generate_suggestions(prompt_text: str, persona_vector: Optional[List[float]] = None, max_suggestions: int = 1) -> List[str]:
#    try:
#        print(f"Generating suggestions for: {prompt_text}", file=sys.stderr, flush=True)
#        # Кодирование промпта
#        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
#
#        # Заглушка для persona_vector
#        if persona_vector is not None:
#            print("Persona vector received, but not integrated in this minimal version.", file=sys.stderr, flush=True)
#
#        # Генерация токенов
#        outputs = model.generate(
#            **inputs,
#            max_new_tokens=20,  # Сокращено с 50 до 20
#            do_sample=True,
#            temperature=0.8,    # Немного снижен для стабильности
#            top_k=50,           # Уменьшен с 64
#            top_p=0.9,          # Уменьшен с 0.95
#            repetition_penalty=1.2,  # Увеличен для разнообразия
#            num_return_sequences=max_suggestions
#        )
#
#        # Декодирование результатов
#        suggestions = []
#        for output in outputs:
#            generated_text = tokenizer.decode(output, skip_special_tokens=True)
#            suggestion = generated_text[len(prompt_text):].strip().split("\n")[0]
#            if suggestion and suggestion not in suggestions:  # Исключение дубликатов
#                suggestions.append(suggestion)
#                
#        print(f"Generated suggestions: {suggestions}", file=sys.stderr, flush=True)
#
#        return suggestions[:max_suggestions]
#
#    except Exception as e:
#        print(f"Error during generation: {e}", file=sys.stderr, flush=True)
#        traceback.print_exc(file=sys.stderr)
#        return []
#
## Основной цикл обработки
#print("Entering main processing loop.", flush=True)
#current_prompt = None
#interrupted = False
#
#while True:
#    try:
#        readable, _, _ = select.select([sys.stdin], [], [], 0.1)  # Увеличен таймаут до 0.1 для более надёжного чтения
#
#        if readable:
#            new_prompt_line = sys.stdin.readline()
#            if not new_prompt_line:
#                break
#
#            new_prompt = new_prompt_line.strip()
#            print(f"Received prompt: {new_prompt}", file=sys.stderr, flush=True)
#
#            if new_prompt:
#                current_prompt = new_prompt
#                interrupted = True
#            else:
#                current_prompt = None
#                interrupted = True
#
#        if current_prompt:
#            prompt_to_process = current_prompt
#            current_prompt = None
#            interrupted = False
#
#            if prompt_to_process:
#                print("Streaming suggestions...", flush=True)
#                suggestions = generate_suggestions(prompt_to_process)
#                
#                for suggestion in suggestions:
#                    if interrupted:
#                        break
#                    print(suggestion, flush=True)
#                    time.sleep(0.01)  # Уменьшена задержка для более быстрого вывода
#                if not interrupted:
#                    print("END_SUGGESTIONS", flush=True)
#            else:
#                print("END_SUGGESTIONS", flush=True)
#
#    except EOFError:
#        break
#    except KeyboardInterrupt:
#        break
#    except Exception as e:
#        print(f"FATAL Error in main loop: {e}", file=sys.stderr, flush=True)
#        traceback.print_exc(file=sys.stderr)
#        break
        
        
#################################### GGUF Model

# Путь к локальной папке модели
here = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(here, "gemma-3-4b-pt-q4_0.gguf")
# Проверка наличия модели
if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}", file=sys.stderr, flush=True)
    sys.exit(1)

# Инициализация модели
try:
    # Подавляем все логи загрузки модели
    with suppress_stderr():
        llm = Llama(model_path=model_path, n_ctx=512, n_threads=8, verbose=False)
    
    print("Model loaded successfully", file=sys.stderr, flush=True)
except Exception as e:
    print(f"ERROR loading model: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

def generate_suggestions(prompt_text: str, persona_vector: Optional[List[float]] = None, max_suggestions: int = 1) -> List[str]:
    if not prompt_text.strip():
        return []
    try:
        # Логируем промпт на входе
        print(f"PROMPT: {prompt_text}", file=sys.stderr, flush=True)
        
        suggestions = []
        for _ in range(max_suggestions):
            output = llm(
                prompt_text,
                max_tokens=15,
                temperature=1.0,
                top_k=64,  # Рекомендация от Unsloth для Gemma 3
                top_p=0.95,
                min_p=0.0,
                repeat_penalty=1.0,
                stop=["<end_of_turn>", "\n"]
            )
            suggestion = output["choices"][0]["text"].strip().split("\n")[0]
            if suggestion and suggestion not in suggestions:
                # Логируем каждый выходной токен/предложение
                print(f"OUTPUT: {suggestion}", file=sys.stderr, flush=True)
                suggestions.append(suggestion)
        print(f"Generated suggestions: {suggestions}", file=sys.stderr, flush=True)
        return suggestions
    except Exception as e:
        print(f"Error during generation: {e}", file=sys.stderr, flush=True)
        return []
    
# Основной цикл обработки
print("Entering main processing loop.", flush=True)
current_prompt = None
interrupted = False


# --- Фикс: корректная обработка прерывания генерации ---
while True:
    try:
        readable, _, _ = select.select([sys.stdin], [], [], 0.05)

        if readable:
            new_prompt_line = sys.stdin.readline()
            if not new_prompt_line:
                break

            new_prompt = new_prompt_line.strip()
            print(f"Received prompt: {new_prompt}", file=sys.stderr, flush=True)

            if new_prompt:
                current_prompt = new_prompt
                interrupted = True  # <-- выставляем флаг прерывания
            else:
                current_prompt = None
                interrupted = True

        if current_prompt:
            prompt_to_process = current_prompt
            current_prompt = None
            interrupted = False  # <-- Сброс сразу после взятия prompt

            if prompt_to_process:
                print("Streaming suggestions...", flush=True)
                suggestions = generate_suggestions(prompt_to_process)
                for suggestion in suggestions:
                    if interrupted:
                        print("INTERRUPTED", file=sys.stderr, flush=True)
                        break
                    words = suggestion.split()
                    for word in words:
                        if interrupted:
                            print("INTERRUPTED", file=sys.stderr, flush=True)
                            break
                        print(word + " ", flush=True)
                        time.sleep(0.07)

                if not interrupted:
                    print("END_SUGGESTIONS", flush=True)
            else:
                print("END_SUGGESTIONS", flush=True)

    except EOFError:
        break
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"FATAL Error in main loop: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        break
        
        
#class MiniLLM(nn.Module):
#    def __init__(self, vocab_size=1000, d_model=128, n_layers=2):
#        super().__init__()
#        self.embed = nn.Embedding(vocab_size, d_model)
#        self.layers = nn.ModuleList([
#            nn.TransformerEncoderLayer(d_model, nhead=4) for _ in range(n_layers)
#        ])
#        self.fc = nn.Linear(d_model, vocab_size)
#
#    def forward(self, input_ids):
#        x = self.embed(input_ids)
#        for layer in self.layers:
#            x = layer(x)
#        return self.fc(x)
#
#here = os.path.dirname(__file__)
#model_path = os.path.join(here, "minillm_export.pt")
#
## Check if model file exists before trying to load it
#if not os.path.exists(model_path):
#    print(f"ERROR: Model file not found at {model_path}", file=sys.stderr, flush=True)
#    print(f"Current directory: {here}", file=sys.stderr, flush=True)
#    print(f"Available files: {os.listdir(here) if os.path.exists(here) else 'Directory not found'}", file=sys.stderr, flush=True)
#    sys.exit(1)
#
#try:
#    print(f"Loading model from: {model_path}", file=sys.stderr, flush=True)
#    model = torch.export.load(model_path)
#    runnable_model = model.module()
#    print("Model loaded successfully", file=sys.stderr, flush=True)
#except Exception as e:
#    print(f"ERROR loading model: {e}", file=sys.stderr, flush=True)
#    traceback.print_exc(file=sys.stderr)
#    sys.exit(1)
#
#vocab = [f"word{i}" for i in range(1000)] # Assuming this dummy vocab is still needed. If not, it can be removed.
#
#print("Entering main processing loop.", flush=True) # KEEP - Essential for Swift to know script is ready
#current_prompt = None
#interrupted = False
#
#while True:
#    try:
#        readable, _, _ = select.select([sys.stdin], [], [], 0.01)
#
#        if readable:
#            new_prompt_line = sys.stdin.readline()
#            if not new_prompt_line:
#                break
#
#            new_prompt = new_prompt_line.strip()
#            if new_prompt:
#                current_prompt = new_prompt
#                interrupted = True
#            else:
#                current_prompt = None
#                interrupted = True
#
#        if current_prompt:
#            prompt_to_process = current_prompt
#            current_prompt = None
#            interrupted = False
#
#            try:
#                tokens = torch.tensor([[ord(c) % 1000 for c in prompt_to_process]], dtype=torch.long)
#
#                with torch.no_grad():
#                    out = runnable_model(tokens)
#                    probs = torch.softmax(out[0, -1], dim=-1)
#                    top5 = torch.topk(probs, 5).indices.tolist()
#
#                print("Streaming suggestions...", flush=True) # KEEP - Essential for Swift
#                max_suggestions = 10 # You might want to make this configurable or remove if not needed
#                output_count = 0
#                for token_id in top5:
#                    if interrupted or output_count >= max_suggestions:
#                        break
#
#                    if 0 <= token_id < len(vocab):
#                        word = vocab[token_id]
#                        print(word, flush=True) # KEEP - Actual suggestion, ensure flush
#                        output_count += 1
#                        time.sleep(0.1) # Keep for simulated streaming if desired
#                    else:
#                        print(f"Warning: Predicted token_id {token_id} out of range.", file=sys.stderr, flush=True)
#
#                if not interrupted:
#                     print("END_SUGGESTIONS", flush=True) # KEEP - Essential for Swift
#
#            except Exception as e:
#                print(f"Error during prediction/streaming for \'{prompt_to_process}\'': {e}", file=sys.stderr, flush=True)
#                traceback.print_exc(file=sys.stderr)
#
#        # Небольшая пауза, чтобы не загружать CPU на 100% в ожидании
#        # time.sleep(0.01) # Можно убрать, если select.select с таймаутом достаточно
#
#    except EOFError:
#        break
#    except KeyboardInterrupt:
#        break
#    except Exception as e:
#        print(f"FATAL Error in main loop: {e}", file=sys.stderr, flush=True)
#        traceback.print_exc(file=sys.stderr)
#        break

#while True:
#    try:
#        readable, _, _ = select.select([sys.stdin], [], [], 0.01)
#        if readable:
#            new_prompt_line = sys.stdin.readline()
#
#    prompt = sys.stdin.readline().strip()
#    if not prompt_line:
#        print("Stdin closed, exiting loop.")
#        break
#    print(f"Prompt received: {prompt}")
#
#    tokens = torch.tensor([[ord(c) % 1000 for c in prompt]], dtype=torch.long)
#    print(f"Converted prompt to tokens: {tokens}")
#
#    print("Making prediction...")
#    with torch.no_grad():
#        out = runnable_model(tokens)
#        probs = torch.softmax(out[0, -1], dim=-1)
#        top5 = torch.topk(probs, 5).indices.tolist()
#
#        print(f"Top 5 predictions: {top5}")
#
#        for token_id in top5:
#            word = vocab[token_id]
#            print(f"Suggested word: {word}")
#            print(word)

##################################

