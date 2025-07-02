from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from huggingface_hub import login


login("HF_LOGIN")

model_name = "google/gemma-3-4b-pt"

embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")

tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-3-4b-pt",
    token="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"
)

llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def encode(text: str) -> np.ndarray:
    return embedder.encode(text)

def compute_personal_embedding(user_history, input_text):
    input_vec = encode(input_text)
    history_vecs = [encode(h) for h in user_history]
    sims = np.array([np.dot(input_vec, h) for h in history_vecs])
    weights = np.exp(sims) / np.sum(np.exp(sims))
    return np.sum([w * h for w, h in zip(weights, history_vecs)], axis=0)

def embed_to_prompt(vector, n_tokens=20):
    vector = vector[:n_tokens]
    return " ".join([f"<E{i}:{v:.4f}>" for i, v in enumerate(vector)])

user_history = [
    "I enjoy writing technical blogs about Python and machine learning.",
    "My favorite tools are PyTorch and Hugging Face transformers.",
    "I often explain concepts in simple language with examples."
]

input_text = "Write a short introduction to BERT for beginners."
persona_vector = compute_personal_embedding(user_history, input_text)
persona_prefix = embed_to_prompt(persona_vector)

full_prompt = f"{persona_prefix}\n\n{input_text}"
inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
outputs = llm.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
  
