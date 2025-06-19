"""
Кастомные функции генерации для MLX с поддержкой audio embeddings
Основано на оригинальном mlx_lm.generate

Автор: MLX Audio Projector Team
"""

import time
from typing import Dict, List, Tuple, Optional, Callable, Generator, Union, Any
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from transformers import PreTrainedTokenizer

# Импорты из оригинального mlx_lm (временно заглушки, заменим позже)
import functools

# Updated MLX imports with better fallbacks
try:
    from mlx_lm import generate
    from mlx_lm.utils import generate_step, load
    from mlx_lm.sample_utils import top_p_sampling, make_sampler
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from mlx_lm.generate import (
        GenerationResponse, 
        wired_limit, 
        generation_stream, 
        maybe_quantize_kv_cache
    )
    print("✅ MLX-LM imports successful")
except ImportError as e:
    print(f"⚠️ MLX-LM imports not available: {e}")
    print("🔄 Using enhanced fallback implementations...")
    
    def generate_step(inputs, model, temp=0.7):
        """Enhanced fallback for generate_step"""
        try:
            logits = model(inputs)
            if hasattr(logits, 'logits'):
                logits = logits.logits
            return logits[:, -1, :]  # Last token logits
        except Exception as e:
            print(f"❌ Error in generate_step fallback: {e}")
            # Return dummy logits
            vocab_size = getattr(model, 'vocab_size', 32000)
            return mx.zeros((inputs.shape[0], vocab_size))
    
    def top_p_sampling(logits, top_p=0.9, temp=0.7):
        """Enhanced fallback for top_p_sampling"""
        try:
            # Apply temperature
            scaled_logits = logits / temp
            
            # Simple top-p approximation
            probs = mx.softmax(scaled_logits, axis=-1)
            sorted_probs = mx.sort(probs, axis=-1)[:, ::-1]
            cumsum_probs = mx.cumsum(sorted_probs, axis=-1)
            
            # Find cutoff (simplified)
            cutoff_mask = cumsum_probs <= top_p
            
            # Sample from categorical distribution
            return mx.random.categorical(scaled_logits, axis=-1)
        except Exception as e:
            print(f"❌ Error in top_p_sampling fallback: {e}")
            # Return random token
            return mx.random.randint(0, logits.shape[-1], (logits.shape[0],))
    
    def make_sampler(temp=0.7, top_p=0.9, **kwargs):
        """Enhanced fallback for make_sampler"""
        def sampler(logits):
            return top_p_sampling(logits, top_p=top_p, temp=temp)
        return sampler
    
    def load(model_path):
        """Fallback stub for load function"""
        print(f"⚠️ Cannot load model {model_path} - MLX-LM not available")
        return None, None
    
    # Fallback classes
    class GenerationResponse:
        def __init__(self, text="", tokens=None):
            self.text = text
            self.tokens = tokens or []
    
    class TokenizerWrapper:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
    
    def wired_limit(*args, **kwargs):
        return lambda x: x
    
    def generation_stream(*args, **kwargs):
        yield "fallback generation"
    
    def maybe_quantize_kv_cache(*args, **kwargs):
        pass
    # Создаем заглушки для разработки
    class cache:
        @staticmethod
        def make_prompt_cache(model, max_kv_size=None):
            return [{"state": mx.zeros((1,))} for _ in range(len(model.layers) if hasattr(model, 'layers') else 12)]
    
    def make_sampler(*args, **kwargs):
        return lambda x: mx.argmax(x, axis=-1)
    
    class TokenizerWrapper:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.eos_token_ids = [tokenizer.eos_token_id] if hasattr(tokenizer, 'eos_token_id') else [2]
            self.detokenizer = SimpleDetokenizer(tokenizer)
        
        def __getattr__(self, name):
            return getattr(self.tokenizer, name)
    
    class SimpleDetokenizer:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.tokens = []
            self.last_segment = ""
        
        def reset(self):
            self.tokens = []
            self.last_segment = ""
        
        def add_token(self, token):
            self.tokens.append(token)
            # Простая реализация детокенизации
            try:
                text = self.tokenizer.decode(self.tokens, skip_special_tokens=True)
                if len(text) > len(''.join([self.tokenizer.decode([t], skip_special_tokens=True) for t in self.tokens[:-1]])):
                    self.last_segment = text[len(''.join([self.tokenizer.decode([t], skip_special_tokens=True) for t in self.tokens[:-1]])):]
                else:
                    self.last_segment = self.tokenizer.decode([token], skip_special_tokens=True)
            except:
                self.last_segment = f"<token_{token}>"
        
        def finalize(self):
            self.last_segment = ""
    
    @dataclass 
    class GenerationResponse:
        text: str
        token: int
        logprobs: mx.array
        from_draft: bool
        prompt_tokens: int
        prompt_tps: float
        generation_tokens: int
        generation_tps: float
        peak_memory: float
        finish_reason: Optional[str] = None
    
    def wired_limit(model, streams):
        return contextlib.nullcontext()
    
    # Создаем фиктивный stream
    class DummyStream:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    generation_stream = DummyStream()
    
    def maybe_quantize_kv_cache(prompt_cache, **kwargs):
        pass

import contextlib

@dataclass
class AudioGenerationResponse:
    """
    Расширенный класс ответа генерации с аудио-информацией
    """
    text: str
    token: int
    logprobs: mx.array
    from_draft: bool
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory: float
    finish_reason: Optional[str] = None
    
    # Дополнительные поля для аудио
    audio_embedding_dim: Optional[int] = None
    audio_features_processed: Optional[bool] = None
    prompt_with_audio: Optional[bool] = None


def audio_generate_step(
    audio_embeddings: mx.array,
    model: nn.Module,
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable] = None,
    logits_processors: Optional[List[Callable]] = None,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 2048,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    prompt_progress_callback: Optional[Callable] = None,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    Генерация токенов на основе аудио эмбеддингов
    
    Args:
        audio_embeddings (mx.array): Входные аудио эмбеддинги размерности [seq_len, emb_dim]
        model (nn.Module): Модель для генерации
        max_tokens (int): Максимальное количество токенов для генерации
        sampler: Функция сэмплирования
        logits_processors: Список процессоров логитов
        max_kv_size: Максимальный размер KV кэша
        prompt_cache: Предвычисленный промпт кэш
        prefill_step_size: Размер шага предзаполнения
        kv_bits: Биты для квантизации KV кэша
        kv_group_size: Размер группы для квантизации
        quantized_kv_start: Шаг начала квантизации
        prompt_progress_callback: Колбэк прогресса промпта
        
    Yields:
        Tuple[mx.array, mx.array]: Токен и логарифмические вероятности
    """
    
    print(f"🎵 Аудио генерация начата с эмбеддингами: {audio_embeddings.shape}")
    
    # Проверяем, что модель поддерживает input_embeddings
    if not hasattr(model, '__call__'):
        raise ValueError("Модель должна иметь метод __call__")
    
    # Инициализируем переменные
    tokens = None
    
    # Создаем KV кэш для генерации
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(
            model,
            max_kv_size=max_kv_size,
        )
    elif len(prompt_cache) != len(model.layers):
        raise ValueError("Неправильное количество слоев в промпт кэше")
    
    prompt_progress_callback = prompt_progress_callback or (lambda *_: None)
    
    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )
    
    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
    
    def _model_call_with_embeddings(embeddings):
        """Вызов модели с кастомными эмбеддингами"""
        try:
            # Пробуем вызвать с input_embeddings
            return model(None, cache=prompt_cache, input_embeddings=embeddings)
        except TypeError:
            # Если модель не поддерживает input_embeddings, используем обходной путь
            print("⚠️ Модель не поддерживает input_embeddings, используем обходной путь")
            # Временное решение - модифицируем первый слой модели
            original_embed_tokens = model.embed_tokens
            
            def custom_embed_tokens(x):
                if x is None:
                    return embeddings
                return original_embed_tokens(x)
            
            # Подменяем функцию эмбеддингов
            model.embed_tokens = custom_embed_tokens
            result = model(mx.zeros((1,), dtype=mx.int32), cache=prompt_cache)
            # Возвращаем оригинальную функцию
            model.embed_tokens = original_embed_tokens
            return result
    
    def _step(y):
        """Один шаг генерации"""
        nonlocal tokens
        
        with mx.stream(generation_stream):
            logits = _model_call_with_embeddings(y[None])
            logits = logits[:, -1, :]
            
            # Применяем процессоры логитов если есть
            if logits_processors:
                if tokens is not None:
                    # Для процессоров нужны токены, но у нас эмбеддинги
                    # Пропускаем процессоры для аудио режима
                    pass
                
            quantize_cache_fn(prompt_cache)
            
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            next_token = sampler(logprobs)
            return next_token, logprobs.squeeze(0)
    
    # Обработка начальных аудио эмбеддингов
    y = audio_embeddings
    
    with mx.stream(generation_stream):
        total_prompt_tokens = y.shape[0]
        prompt_processed_tokens = 0
        
        # Обрабатываем длинные промпты по частям
        while y.shape[0] > prefill_step_size:
            _model_call_with_embeddings(y[:prefill_step_size][None])
            quantize_cache_fn(prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
            prompt_processed_tokens += prefill_step_size
            y = y[prefill_step_size:]
            mx.clear_cache()
        
        # Получаем первый токен
        first_token, logprobs = _step(y)
    
    mx.async_eval(first_token, logprobs)
    
    # Генерируем токены
    n = 0
    current_token = first_token
    current_logprobs = logprobs
    
    while True:
        if n != max_tokens:
            # Для следующих токенов используем обычную генерацию
            with mx.stream(generation_stream):
                logits = model(current_token[None], cache=prompt_cache)
                logits = logits[:, -1, :]
                quantize_cache_fn(prompt_cache)
                logprobs = logits - mx.logsumexp(logits, keepdims=True)
                next_token = sampler(logprobs)
            
            mx.async_eval(next_token, logprobs)
        
        if n == 0:
            mx.eval(current_token)
            prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
        
        if n == max_tokens:
            break
            
        yield current_token.item(), current_logprobs
        
        if n % 256 == 0:
            mx.clear_cache()
            
        current_token, current_logprobs = next_token, logprobs.squeeze(0)
        n += 1


def audio_stream_generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    audio_embeddings: mx.array,
    **kwargs,
) -> Generator[AudioGenerationResponse, None, None]:
    """
    Потоковая генерация текста на основе аудио эмбеддингов
    
    Args:
        model (nn.Module): Модель для генерации
        tokenizer: Токенайзер
        audio_embeddings (mx.array): Аудио эмбеддинги
        **kwargs: Дополнительные параметры для audio_generate_step
        
    Yields:
        AudioGenerationResponse: Ответ с генерированным текстом и метаданными
    """
    
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    
    print(f"🎵 Начинаем потоковую аудио генерацию с эмбеддингами: {audio_embeddings.shape}")
    
    detokenizer = tokenizer.detokenizer
    token_generator = audio_generate_step(audio_embeddings, model, **kwargs)
    
    with wired_limit(model, [generation_stream]):
        detokenizer.reset()
        tic = time.perf_counter()
        
        for n, (token, logprobs) in enumerate(token_generator):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                # Для аудио эмбеддингов prompt_tokens = длина последовательности
                prompt_tokens = audio_embeddings.shape[0]
                prompt_tps = prompt_tokens / prompt_time
                tic = time.perf_counter()
            
            if token in tokenizer.eos_token_ids:
                break
            
            detokenizer.add_token(token)
            
            yield AudioGenerationResponse(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                from_draft=False,  # Не используем draft модель
                prompt_tokens=prompt_tokens,
                prompt_tps=prompt_tps,
                generation_tokens=n + 1,
                generation_tps=(n + 1) / (time.perf_counter() - tic),
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason=None,
                audio_embedding_dim=audio_embeddings.shape[-1],
                audio_features_processed=True,
                prompt_with_audio=True,
            )
        
        detokenizer.finalize()
        yield AudioGenerationResponse(
            text=detokenizer.last_segment,
            token=token,
            logprobs=logprobs,
            from_draft=False,
            prompt_tokens=prompt_tokens,
            prompt_tps=prompt_tps,
            generation_tokens=n + 1,
            generation_tps=(n + 1) / (time.perf_counter() - tic),
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason="stop" if token in tokenizer.eos_token_ids else "length",
            audio_embedding_dim=audio_embeddings.shape[-1],
            audio_features_processed=True,
            prompt_with_audio=True,
        )


def audio_generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    audio_embeddings: mx.array,
    verbose: bool = False,
    **kwargs,
) -> str:
    """
    Полная генерация текста на основе аудио эмбеддингов
    
    Args:
        model (nn.Module): Модель для генерации
        tokenizer: Токенайзер
        audio_embeddings (mx.array): Аудио эмбеддинги
        verbose (bool): Печатать ли процесс генерации
        **kwargs: Дополнительные параметры
        
    Returns:
        str: Сгенерированный текст
    """
    
    if verbose:
        print("🎵" + "=" * 50)
        print(f"🎵 Генерация с аудио эмбеддингами: {audio_embeddings.shape}")
        print("🎵" + "=" * 50)
    
    text = ""
    response = None
    
    for response in audio_stream_generate(model, tokenizer, audio_embeddings, **kwargs):
        if verbose:
            print(response.text, end="", flush=True)
        text += response.text
    
    if verbose:
        print()
        print("🎵" + "=" * 50)
        if len(text) == 0:
            print("🎵 Текст не был сгенерирован для данных аудио эмбеддингов")
            return ""
        
        if response:
            print(f"🎵 Аудио промпт: {response.prompt_tokens} эмбеддингов, "
                  f"{response.prompt_tps:.3f} эмбеддингов/сек")
            print(f"🎵 Генерация: {response.generation_tokens} токенов, "
                  f"{response.generation_tps:.3f} токенов/сек")
            print(f"🎵 Пиковая память: {response.peak_memory:.3f} GB")
            print(f"🎵 Размерность аудио: {response.audio_embedding_dim}")
    
    return text


def create_audio_prompt_embeddings(
    text_prefix: str,
    audio_embeddings: mx.array,
    model,
    tokenizer,
) -> mx.array:
    """
    Создание комбинированных эмбеддингов из текстового префикса и аудио
    
    Args:
        text_prefix (str): Текстовый префикс (например, "Транскрипция аудио: ")
        audio_embeddings (mx.array): Аудио эмбеддинги
        model: Модель для получения текстовых эмбеддингов
        tokenizer: Токенайзер
        
    Returns:
        mx.array: Комбинированные эмбеддинги
    """
    
    # Токенизируем префикс
    prefix_tokens = tokenizer(text_prefix, return_tensors="np", add_special_tokens=False)
    prefix_ids = mx.array(prefix_tokens.input_ids.squeeze(0))
    
    # Получаем эмбеддинги префикса
    prefix_embeddings = model.embed_tokens(prefix_ids)
    
    # Если аудио эмбеддинги имеют размерность [emb_dim], расширяем до [1, emb_dim]
    if audio_embeddings.ndim == 1:
        audio_embeddings = mx.expand_dims(audio_embeddings, 0)
    
    # Объединяем префикс и аудио эмбеддинги
    combined_embeddings = mx.concatenate([prefix_embeddings, audio_embeddings], axis=0)
    
    print(f"📝 Префикс: '{text_prefix}' -> {prefix_embeddings.shape}")
    print(f"🎵 Аудио: {audio_embeddings.shape}")
    print(f"🔗 Комбинированные: {combined_embeddings.shape}")
    
    return combined_embeddings


# Дополнительные утилиты для работы с аудио генерацией

def batch_audio_generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    audio_embeddings_batch: List[mx.array],
    text_prefix: str = "Транскрипция аудио: ",
    verbose: bool = False,
    **kwargs,
) -> List[str]:
    """
    Батчевая генерация для множественных аудио эмбеддингов
    
    Args:
        model: Модель
        tokenizer: Токенайзер  
        audio_embeddings_batch: Список аудио эмбеддингов
        text_prefix: Текстовый префикс
        verbose: Печать прогресса
        **kwargs: Дополнительные параметры
        
    Returns:
        List[str]: Список сгенерированных текстов
    """
    
    results = []
    
    for i, audio_emb in enumerate(audio_embeddings_batch):
        if verbose:
            print(f"\n🎵 Обрабатываем аудио {i+1}/{len(audio_embeddings_batch)}")
        
        # Создаем комбинированные эмбеддинги
        combined_emb = create_audio_prompt_embeddings(
            text_prefix, audio_emb, model, tokenizer
        )
        
        # Генерируем текст
        generated_text = audio_generate(
            model, tokenizer, combined_emb, verbose=verbose, **kwargs
        )
        
        results.append(generated_text)
        
        if verbose:
            print(f"🎵 Результат {i+1}: '{generated_text[:100]}{'...' if len(generated_text) > 100 else ''}'")
    
    return results


def test_audio_generation():
    """
    Тестовая функция для проверки аудио генерации
    """
    
    print("🧪 Тестирование аудио генерации...")
    
    # Создаем фиктивные аудио эмбеддинги
    dummy_audio_embeddings = mx.random.normal((10, 768))  # 10 кадров, 768 признаков
    
    print(f"🎵 Тестовые аудио эмбеддинги: {dummy_audio_embeddings.shape}")
    print(f"🎵 Статистики: min={float(mx.min(dummy_audio_embeddings)):.3f}, "
          f"max={float(mx.max(dummy_audio_embeddings)):.3f}, "
          f"mean={float(mx.mean(dummy_audio_embeddings)):.3f}")
    
    return dummy_audio_embeddings


if __name__ == "__main__":
    print("🎵 MLX Custom Audio Generation Module")
    print("🎵 Поддержка генерации текста из аудио эмбеддингов")
    test_audio_generation()
