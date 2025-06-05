import torch
import sys
import os
import traceback
import select
import time
import torch.nn as nn
import torch.nn.functional as F

# Disable Metal Performance Shaders to avoid mach-O errors in app bundle
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Force CPU backend to avoid GPU-related issues in packaged app
torch.backends.mps.is_available = lambda: False
torch.backends.cuda.is_available = lambda: False

class MiniLLM(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead=4) for _ in range(n_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.fc(x)

here = os.path.dirname(__file__)
model_path = os.path.join(here, "minillm_export.pt")

# Check if model file exists before trying to load it
if not os.path.exists(model_path):
    print(f"ERROR: Model file not found at {model_path}", file=sys.stderr, flush=True)
    print(f"Current directory: {here}", file=sys.stderr, flush=True)
    print(f"Available files: {os.listdir(here) if os.path.exists(here) else 'Directory not found'}", file=sys.stderr, flush=True)
    sys.exit(1)

try:
    print(f"Loading model from: {model_path}", file=sys.stderr, flush=True)
    model = torch.export.load(model_path)
    runnable_model = model.module()
    print("Model loaded successfully", file=sys.stderr, flush=True)
except Exception as e:
    print(f"ERROR loading model: {e}", file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

vocab = [f"word{i}" for i in range(1000)] # Assuming this dummy vocab is still needed. If not, it can be removed.

print("Entering main processing loop.", flush=True) # KEEP - Essential for Swift to know script is ready
current_prompt = None
interrupted = False

while True:
    try:
        readable, _, _ = select.select([sys.stdin], [], [], 0.01)

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

            try:
                tokens = torch.tensor([[ord(c) % 1000 for c in prompt_to_process]], dtype=torch.long)

                with torch.no_grad():
                    out = runnable_model(tokens)
                    probs = torch.softmax(out[0, -1], dim=-1)
                    top5 = torch.topk(probs, 5).indices.tolist()

                print("Streaming suggestions...", flush=True) # KEEP - Essential for Swift
                max_suggestions = 10 # You might want to make this configurable or remove if not needed
                output_count = 0
                for token_id in top5:
                    if interrupted or output_count >= max_suggestions:
                        break

                    if 0 <= token_id < len(vocab):
                        word = vocab[token_id]
                        print(word, flush=True) # KEEP - Actual suggestion, ensure flush
                        output_count += 1
                        time.sleep(0.1) # Keep for simulated streaming if desired
                    else:
                        print(f"Warning: Predicted token_id {token_id} out of range.", file=sys.stderr, flush=True)

                if not interrupted:
                     print("END_SUGGESTIONS", flush=True) # KEEP - Essential for Swift

            except Exception as e:
                print(f"Error during prediction/streaming for \'{prompt_to_process}\'': {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)

        # Небольшая пауза, чтобы не загружать CPU на 100% в ожидании
        # time.sleep(0.01) # Можно убрать, если select.select с таймаутом достаточно

    except EOFError:
        break
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"FATAL Error in main loop: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        break

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

#import sys
#import os
#import time # Для имитации задержки, если нужно
#
#print("Python dummy script started.")
#
#try:
#    prompt = sys.stdin.readline() # Все равно читаем ввод, чтобы Swift не завис
#    if prompt:
#        prompt = prompt.strip()
#        print(f"Dummy mode: Prompt received: '{prompt}'")
#
#        # Просто печатаем шаблонный результат
#        print("template_word_1")
#        # Можно добавить несколько для теста
#        print("another_suggestion")
#
#        # time.sleep(1) # Можно раскомментировать для имитации долгой работы
#
#    else:
#        print("Dummy mode: No prompt received on stdin.")
#
#except Exception as e:
#     print(f"Dummy mode: Error during processing: {e}", file=sys.stderr)
#
#print("Python dummy script finished.")
