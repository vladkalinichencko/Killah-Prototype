from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer
import json

login("HF_AUTHORIZATION")

dataset = load_dataset(
    "institutional/institutional-books-1.0", split="train", streaming=True)

tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-3-4b-pt",
    token="HF_TOKENIZER"
)

output_file = "long_books_ru.jsonl"
min_tokens = 128_000
saved_count = 0
max_books = 100


def clean_text(text):
    text = text.replace('\n', ' ').strip()
    return text


with open(output_file, "w", encoding="utf-8") as f:
    for example in dataset:
        if saved_count >= max_books:
            break

        lang = example.get("language_gen") or example.get("language_src")
        if lang.lower() not in {"ru", "rus", "russ", "russian"}:
            continue

        pages = example.get("text_by_page_gen") or example.get(
            "text_by_page_src")
        if not pages or not isinstance(pages, list):
            continue

        full_text = " ".join(pages)
        full_text = clean_text(full_text)

        tokens = tokenizer(full_text, return_attention_mask=False,
                           return_token_type_ids=False).input_ids

        if len(tokens) >= min_tokens:
            json.dump({
                "title": example.get("title_src", "untitled"),
                "author": example.get("author_src", "unknown"),
                "text": full_text
            }, f, ensure_ascii=False)
            f.write("\n")
            saved_count += 1

print(f"🎯 Готово! Сохранено книг: {saved_count}")
