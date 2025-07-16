import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Стоп-слова для обоих языков
STOP_WORDS = {
    'ru': {
        'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 
        'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 
        'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 
        'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 
        'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 
        'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 
        'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 
        'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 
        'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 
        'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 
        'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 
        'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 
        'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 
        'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 
        'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 
        'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 
        'более', 'всегда', 'конечно', 'всю', 'между'
    },
    'en': {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
        'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
        'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 
        'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
        'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
        'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
        'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
        'with', 'about', 'against', 'between', 'into', 'through', 'during', 
        'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 
        'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 
        'will', 'just', 'should', 'now'
    }
}

def preprocess_text(text):
    """Простая предобработка текста для обоих языков"""
    # Приводим к нижнему регистру
    text = text.lower()
    
    # Удаляем пунктуацию и специальные символы
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Удаляем цифры
    text = re.sub(r'\d+', '', text)
    
    # Токенизация (разбиение на слова)
    words = text.split()
    
    # Удаляем стоп-слова для обоих языков
    words = [word for word in words if word not in STOP_WORDS['ru'] and word not in STOP_WORDS['en']]
    
    return ' '.join(words)

def cosine_similarity_attention(target, histories):
    """
    Вычисляет косинусное сходство между target и каждым элементом в histories.
    Работает с русским и английским языками.
    """
    # Объединяем строки в каждом элементе
    target_text = ' '.join(target)
    histories_texts = [' '.join(history) for history in histories]
    
    # Предобработка текстов
    target_processed = preprocess_text(target_text)
    histories_processed = [preprocess_text(text) for text in histories_texts]
    
    # Создаем TF-IDF векторайзер (работает с любыми языками)
    vectorizer = TfidfVectorizer()
    
    # Объединяем все тексты для обучения векторайзера
    all_texts = [target_processed] + histories_processed
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Разделяем обратно на target и histories
    target_vector = tfidf_matrix[0]
    histories_vectors = tfidf_matrix[1:]
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
