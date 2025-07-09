from pathlib import Path
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer
import json
import re

login("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")

dataset = load_dataset(
    "institutional/institutional-books-1.0", split="train", streaming=True)

tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-3-4b-pt",
    token="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"
)

output_file = Path(
    "C:/Users/serma/killah_project/datasets/TextDatasets/BooksDatasets/long_books_eng_cleaned.jsonl")
min_tokens = 5000
max_books = 5000
saved_count = 0


def clean_text(text: str, top_cut: float = 0.45, bottom_cut: float = 0.49) -> str:
    n = len(text)
    text = text[int(n * top_cut): int(n * (1 - bottom_cut))]
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s{2,}", " ", text)
    typographic_garbage = [
        "•", "◊", "◆", "□", "■", "▶", "◦", "▪", "►", "※",
        "™", "©", "®", "—", "–", "―", "·", "…", "<<", ">>",
        "†", "‡", "§", "¶", "~", "¤", "°", "º", "×"
    ]
    for symbol in typographic_garbage:
        text = text.replace(symbol, "")
    text = re.sub(r"[^\x00-\x7Fа-яА-ЯёЁ0-9 .,!?\"'():;\\/-]", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


with open(output_file, "w", encoding="utf-8") as f:
    for example in dataset:
        if saved_count >= max_books:
            break

        lang = example.get("language_gen") or example.get("language_src")
        if not lang or lang.lower() not in {"en", "eng", "english"}:
            continue

        pages = example.get("text_by_page_gen") or example.get(
            "text_by_page_src")
        if not pages or not isinstance(pages, list):
            continue

        full_text = " ".join(pages)
        cleaned_text = clean_text(full_text)

        tokens = tokenizer(cleaned_text, return_attention_mask=False,
                           return_token_type_ids=False).input_ids

        if len(tokens) >= min_tokens:
            json.dump({
                "title": example.get("title_src", "untitled"),
                "author": example.get("author_src", "unknown"),
                "text": cleaned_text
            }, f, ensure_ascii=False)
            f.write("\n")
            saved_count += 1

print(f"🎯 Готово! Сохранено книг: {saved_count}")
