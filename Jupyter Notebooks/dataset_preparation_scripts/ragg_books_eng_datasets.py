import json
import random
from pathlib import Path
from transformers import AutoTokenizer
import re


# 📁 Настройки
input_path = Path(
    "C:/Users/serma/killah_project/datasets/TextDatasets/BooksDatasets/long_books_eng_cleaned.jsonl")
output_path = Path(
    "C:/Users/serma/killah_project/datasets/TextDatasets/BooksDatasets/ragged_books_eng.jsonl")

min_len = 1000
max_len = 100_000
fragments_per_book = 100  # можно настроить
model_name = "google/gemma-3-4b-pt"
token = "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"

# 🔤 Токенизатор
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
tokenizer.model_max_length = 1_000_000  # разрешаем длину токенов явно
tokenizer.truncation_side = "right"


def decode_fragment(token_ids):
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def slice_fragment(input_ids):
    total_len = len(input_ids)
    frag_len = random.randint(min_len, min(max_len, total_len // 2))

    if total_len < frag_len + 10:
        return None

    start = random.randint(0, total_len - frag_len)
    end = start + frag_len

    frag = input_ids[start:end]

    # Типы разрезов
    mode = random.choice(["middle_word", "start_word", "start_sentence"])

    if mode == "middle_word":
        # ничего не делаем — сохраняем разрез как есть
        pass

    elif mode == "start_word":
        # найти границу между словами и обрезать до неё
        decoded = tokenizer.decode(frag, skip_special_tokens=True)
        match = next((m.start() for m in re.finditer(r"\b\w", decoded)), None)
        if match:
            new_frag = tokenizer(
                decoded[match:], return_tensors="pt").input_ids[0]
            frag = new_frag if len(new_frag) >= min_len else frag

    elif mode == "start_sentence":
        decoded = tokenizer.decode(frag, skip_special_tokens=True)
        sentences = re.split(r'(?<=[.!?])\s+', decoded)
        if len(sentences) >= 2:
            chosen = " ".join(sentences[1:])
            new_frag = tokenizer(chosen, return_tensors="pt").input_ids[0]
            frag = new_frag if len(new_frag) >= min_len else frag

    return frag


# 📦 Чтение и нарезка
with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
    for line in fin:
        book = json.loads(line)
        text = book.get("text", "")
        if not text or len(text) < 1000:
            continue

        input_ids = tokenizer(text, return_attention_mask=False,
                              return_token_type_ids=False).input_ids
        if len(input_ids) < min_len:
            continue

        title = book.get("title", "unknown")
        author = book.get("author", "unknown")

        for _ in range(fragments_per_book):
            fragment = slice_fragment(input_ids)
            if fragment is not None and len(fragment) > 0:
                fout.write(json.dumps({
                    "title": title,
                    "author": author,
                    "text": decode_fragment(fragment),
                    "num_tokens": len(fragment)
                }, ensure_ascii=False) + "\n")

print("🎯 Нарезка завершена. Файл сохранён:", output_path)
