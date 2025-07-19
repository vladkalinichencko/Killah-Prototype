#!/usr/bin/env python3
"""
Модульная инъекция эмбеддингов в GGUF модель
- Персонализация: один токен в начале
- Аудио: массив токенов в конце
- Текст: между ними
"""
import os
import time
import numpy as np
from llama_cpp import (
    Llama, llama_batch_init, llama_batch_add_embedding, 
    llama_batch_add_token, llama_batch_free, llama_decode, 
    llama_get_logits_ith
)

class EmbeddingInjector:
    def __init__(self, model_path, n_ctx=2048):
        """Инициализация модели"""
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        print("Загрузка модели...")
        self.llm = Llama(
            model_path=model_path, 
            n_ctx=n_ctx, 
            embedding=True, 
            n_gpu_layers=-1, 
            verbose=False
        )
        self.n_embd = self.llm.n_embd()
        print(f"Размерность эмбеддингов: {self.n_embd}")
    
    def create_random_embedding(self):
        """Создает случайный эмбеддинг"""
        return np.random.randn(self.n_embd).astype(np.float32)
    
    def create_personalization_token(self):
        """Создает токен персонализации (один эмбеддинг)"""
        return self.create_random_embedding()
    
    def create_audio_tokens(self, count=3):
        """Создает аудио-токены (массив эмбеддингов)"""
        return [self.create_random_embedding() for _ in range(count)]
    
    def inject_and_infer(self, text_prompt, personalization_emb=None, audio_embs=None):
        """
        Основная функция инференса с инъекцией эмбеддингов
        Порядок: [персонализация] + текст + [аудио-токены]
        """
        # Подготовка эмбеддингов
        if personalization_emb is None:
            personalization_emb = self.create_personalization_token()
        
        if audio_embs is None:
            audio_embs = self.create_audio_tokens()
        
        # Токенизация текста
        text_tokens = self.llm.tokenize(text_prompt.encode("utf-8"))
        
        # Подсчет общего количества токенов
        n_pers = 1  # один токен персонализации
        n_audio = len(audio_embs)  # количество аудио-токенов
        n_text = len(text_tokens)
        total_tokens = n_pers + n_text + n_audio
        
        print(f"Токены: персонализация={n_pers}, текст={n_text}, аудио={n_audio}, всего={total_tokens}")
        
        # Создание batch
        batch = llama_batch_init(total_tokens, self.n_embd, 1)
        
        # 1. Добавляем персонализацию в начало
        llama_batch_add_embedding(batch, personalization_emb, 0, 0)
        
        # 2. Добавляем текстовые токены
        for i, token in enumerate(text_tokens):
            llama_batch_add_token(batch, token, n_pers + i, 0)
        
        # 3. Добавляем аудио-токены в конец
        for i, audio_emb in enumerate(audio_embs):
            llama_batch_add_embedding(batch, audio_emb, n_pers + n_text + i, 0)
        
        # Выполнение инференса
        print("Выполнение инференса...")
        start_time = time.perf_counter()
        ret = llama_decode(self.llm.ctx, batch)
        inference_time = (time.perf_counter() - start_time) * 1000
        
        if ret != 0:
            llama_batch_free(batch)
            raise RuntimeError(f"Ошибка декодирования: {ret}")
        
        print(f"Инференс завершен за {inference_time:.2f} мс")
        
        # Получение результатов
        vocab_size = self.llm.n_vocab()
        logits_ptr = llama_get_logits_ith(self.llm.ctx, total_tokens - 1)
        logits = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,))
        
        # Очистка
        llama_batch_free(batch)
        
        return logits
    
    def print_top_tokens(self, logits, top_k=5):
        """Выводит топ-K токенов с их логитами"""
        top_indices = logits.argsort()[-top_k:][::-1]
        print(f"\nТоп-{top_k} токенов:")
        for idx in top_indices:
            token_text = self.llm.detokenize([int(idx)]).decode("utf-8", "ignore")
            print(f"  {idx:6d}: {token_text!r:15} (logit: {logits[idx]:.3f})")
    
    def generate_response(self, text_prompt, personalization_emb=None, audio_embs=None, max_tokens=50):
        """Генерирует ответ модели с инъекцией эмбеддингов"""
        logits = self.inject_and_infer(text_prompt, personalization_emb, audio_embs)
        self.print_top_tokens(logits)
        
        # Можно добавить автоматическую генерацию продолжения
        # Пока просто возвращаем логиты
        return logits

# Конфигурация
MODEL_PATH = (
    "/Users/vladislavkalinichenko/Library/Containers/"
    "com.vladotpad.Killah-Prototype/Data/Library/Application Support/"
    "KillahPrototype/models/gemma/gemma-3-4b-pt-q4_0.gguf"
)

def main():
    """Демонстрация работы"""
    # Инициализация
    injector = EmbeddingInjector(MODEL_PATH)
    
    # Тестовый текст
    text_prompt = "Привет! Как дела? Расскажи что-то интересное"
    
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ ИНЪЕКЦИИ ЭМБЕДДИНГОВ")
    print("="*60)
    
    # Создание кастомных эмбеддингов (пока случайных)
    personalization = injector.create_personalization_token()
    audio_tokens = injector.create_audio_tokens(count=2)  # 2 аудио-токена
    
    print(f"\nТекст: {text_prompt}")
    print(f"Персонализация: вектор размерности {len(personalization)}")
    print(f"Аудио-токены: {len(audio_tokens)} векторов размерности {len(audio_tokens[0])}")
    
    # Выполнение инференса
    logits = injector.generate_response(text_prompt, personalization, audio_tokens)
    
    print("\n✅ Инференс завершен успешно!")

if __name__ == "__main__":
    main() 