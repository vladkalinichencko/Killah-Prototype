#!/usr/bin/env python3
""" 
ACTUAL WORKING EXAMPLE: injecting arbitrary embeddings into a modern GGUF model (llama-cpp-python ≥0.3.x)
The script:
1. Loads the model (Gemma 3-4B in the path below – change if needed)
2. Builds a low-level llama_batch via ctypes helpers
3. Adds two random vectors as "virtual tokens" followed by a text prompt
4. Performs a single forward pass (llama_decode) and prints top-5 logits
Replace `custom_emb` with real audio vectors of length n_embd for experiments.
"""
import os
import ctypes
import time
import numpy as np
from llama_cpp import (
    Llama, llama_cpp,
    llama_batch_init, llama_batch_add_embedding, llama_batch_add_token,
    llama_batch_free, llama_decode, llama_get_logits_ith,
)

# ---------------------------------------------------------------------------
# CONFIG – adjust the path to your GGUF model if necessary
# ---------------------------------------------------------------------------
MODEL_PATH = (
    "/Users/vladislavkalinichenko/Library/Containers/"
    "com.vladotpad.Killah-Prototype/Data/Library/Application Support/"
    "KillahPrototype/models/gemma/gemma-3-4b-pt-q4_0.gguf"
)
TEXT_PROMPT = (
    "Это демонстрация низкоуровневой инъекции эмбеддингов. "
    "Сейчас модель получит два случайных вектора перед текстом."
)
N_VIRTUAL = 2  # how many virtual-embedding tokens we insert

# ---------------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------------
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}")

print("⇒ loading model …")
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, embedding=True, n_gpu_layers=-1, verbose=False)

n_embd = llm.n_embd()
print("n_embd =", n_embd)

# ---------------------------------------------------------------------------
# PREPARE EMBEDDINGS AND TOKENS
# ---------------------------------------------------------------------------
vec1 = np.random.randn(n_embd).astype(np.float32)
vec2 = np.random.randn(n_embd).astype(np.float32)
text_tokens = llm.tokenize(TEXT_PROMPT.encode("utf-8"))

total_tokens = N_VIRTUAL + len(text_tokens)
print("total tokens =", total_tokens)

# ---------------------------------------------------------------------------
# BUILD BATCH (low-level C API)
# ---------------------------------------------------------------------------
# init batch: (n_tokens, embd_size, n_seq_max=1)
batch = llama_batch_init(total_tokens, n_embd, 1)

# add embeddings
llama_batch_add_embedding(batch, vec1, 0, 0)  # pos 0, seq_id 0
llama_batch_add_embedding(batch, vec2, 1, 0)  # pos 1

# add text tokens after virtual embeddings
for i, tok in enumerate(text_tokens):
    llama_batch_add_token(batch, tok, N_VIRTUAL + i, 0)

print("⇒ llama_decode …")
start = time.perf_counter()
ret = llama_decode(llm.ctx, batch)
print(f"decode status = {ret} (%.2f ms)" % ((time.perf_counter() - start) * 1e3))

if ret != 0:
    raise RuntimeError("llama_decode returned error code")

# ---------------------------------------------------------------------------
# INSPECT OUTPUT LOGITS
# ---------------------------------------------------------------------------
vocab = llm.n_vocab()
logits_ptr = llama_get_logits_ith(llm.ctx, total_tokens - 1)  # last token
logits = np.ctypeslib.as_array(logits_ptr, shape=(vocab,))

top5 = logits.argsort()[-5:][::-1]
print("Top-5 tokens after injection:")
for tid in top5:
    txt = llm.detokenize([int(tid)]).decode("utf-8", "ignore")
    print(f"  {tid:6d}: {txt!r}  logit={logits[tid]:.3f}")

# clean
llama_batch_free(batch)
print("✓ done") 