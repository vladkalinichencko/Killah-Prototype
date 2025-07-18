import sys
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity_attention(target, histories):
    """
    Вычисляет косинусное сходство между эмбеддингом target и каждым эмбеддингом в histories.
    """
    # Преобразуем входные данные в numpy массивы
    target_vector = np.array(target, dtype=np.float32).reshape(1, -1)
    histories_vectors = np.array(histories, dtype=np.float32)
    
    # Проверяем, что векторы не пустые и имеют одинаковую размерность
    if target_vector.shape[1] != histories_vectors.shape[1]:
        raise ValueError("Target and histories must have the same embedding dimension")
    if target_vector.size == 0 or histories_vectors.size == 0:
        raise ValueError("Empty vectors provided")
    
    # Вычисляем косинусное сходство
    similarities = cosine_similarity(target_vector, histories_vectors)
    return similarities[0].tolist()

def normalize_attention_weights(similarities):
    """
    Нормализует значения внимания так, чтобы минимальное было 0.1, максимальное 1,
    а остальные пропорционально между ними.
    """
    if not similarities:
        return []
    
    min_sim = min(similarities)
    max_sim = max(similarities)
    
    # Если все значения одинаковые (например, все нули)
    if max_sim == min_sim:
        return [1.0] * len(similarities)
    
    # Нормализация к диапазону [0.1, 1]
    normalized = []
    for sim in similarities:
        norm = 0.1 + 0.9 * (sim - min_sim) / (max_sim - min_sim)
        normalized.append(norm)
    
    return normalized

if __name__ == "__main__":
    print("READY", flush=True)
    while True:
        try:
            input_data = input().strip()
            if input_data:
                data = json.loads(input_data)
                target = data['target']  # Ожидаем список чисел [float]
                histories = data['histories']  # Ожидаем список списков чисел [[float], [float], ...]
                similarities = cosine_similarity_attention(target, histories)
                normalized_weights = normalize_attention_weights(similarities)
                print(json.dumps(normalized_weights), flush=True)
                print("END", flush=True)
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr, flush=True)
