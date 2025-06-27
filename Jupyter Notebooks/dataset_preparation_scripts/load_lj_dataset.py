import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime
from pathlib import Path

USERNAME = "tema"
START_YEAR = 2020
END_YEAR = 2022
MIN_LENGTH = 3000
OUTPUT_FILE = Path(
    f"C:/Users/serma/killah_project/datasets/TextDatasets/LJDatasets/{USERNAME}_lj_posts.jsonl")

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

BASE_URL = f"https://{USERNAME}.livejournal.com"


def get_month_urls():
    urls = []
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            url = f"{BASE_URL}/{year:04d}/{month:02d}/"
            urls.append(url)
    return urls


def extract_post_links(archive_html):
    soup = BeautifulSoup(archive_html, "html.parser")
    return list(set(a['href'] for a in soup.find_all("a", href=True)
                    if a['href'].startswith(BASE_URL) and a['href'].endswith(".html")))


def parse_post(url):
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        # Заголовок
        title_tag = soup.select_one(".entry-title, .asset-title")
        title = title_tag.text.strip() if title_tag else "Без заголовка"

        # Тело поста
        body_tag = soup.select_one(".entry-content, .asset-body")
        if not body_tag:
            return None

        text = body_tag.get_text(separator="\n").strip()
        if len(text) < MIN_LENGTH:
            return None

        # Дата поста (по URL)
        try:
            date_str = url.split("/")[-3:-1]
            date = datetime.strptime(
                "-".join(date_str), "%Y-%m").strftime("%Y-%m")
        except Exception:
            date = ""

        return {
            "author": USERNAME,
            "url": url,
            "title": title,
            "date": date,
            "text": text
        }

    except Exception as e:
        print(f"❌ Ошибка парсинга поста: {url}\n→ {e}")
        return None


def main():
    total_saved = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for month_url in get_month_urls():
            print(f"📅 Читаю архив: {month_url}")
            try:
                res = requests.get(month_url, headers=HEADERS, timeout=10)
                post_urls = extract_post_links(res.text)
                print(f"  🔗 Найдено постов: {len(post_urls)}")

                for post_url in post_urls:
                    post = parse_post(post_url)
                    if post:
                        out.write(json.dumps(post, ensure_ascii=False) + "\n")
                        total_saved += 1
                        print(
                            f"    ✅ Сохранён: {post['title'][:50]}... ({len(post['text'])} символов)")

                    time.sleep(1)

            except Exception as e:
                print(f"⚠️ Ошибка чтения архива: {month_url} — {e}")
                continue

    print(f"\n🎯 Завершено! Всего сохранено постов: {total_saved}")


if __name__ == "__main__":
    main()
