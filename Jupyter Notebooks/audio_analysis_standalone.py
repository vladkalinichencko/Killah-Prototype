#!/usr/bin/env python3
"""
🔬 Автономный анализатор аудио-текстовых embedding'ов
Отдельный файл для анализа обученной модели без обучения
"""

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from typing import List, Tuple, Dict, Optional
import os
import json
from tqdm import tqdm
import wandb

# Дополнительные импорты для полной инициализации модели
from transformers import AutoTokenizer, AutoConfig
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class AudioTextEmbeddingAnalyzer:
    """
    🔬 Исследовательский класс для анализа соответствия между аудио и текстовыми embedding'ами
    ИСПРАВЛЕНО: Все операции переведены в float32 для совместимости
    """
    
    def __init__(self, model, projector, wav2vec2, tokenizer, device, save_to_wandb=True):
        self.model = model
        self.projector = projector
        self.wav2vec2 = wav2vec2
        self.tokenizer = tokenizer
        self.device = device
        self.save_to_wandb = save_to_wandb
        
        # Создаем vocabulary embedding matrix для поиска ближайших токенов
        self._create_vocab_embeddings()
        
    def _create_vocab_embeddings(self):
        """Создает матрицу embedding'ов для всего словаря"""
        print("🔤 Создание матрицы embedding'ов словаря...")
        vocab_size = self.tokenizer.vocab_size
        
        # Берем первые 10000 токенов для ускорения
        max_tokens = min(10000, vocab_size)
        token_ids = torch.arange(max_tokens, device=self.device)
        
        with torch.no_grad():
            # 🔧 ИСПРАВЛЕНИЕ: Принудительно конвертируем в float32
            embeddings = self.model.get_input_embeddings()(token_ids)
            self.vocab_embeddings = embeddings.float()  # ← Конвертация в float32
            
        # Создаем mapping token_id -> token_text
        self.token_id_to_text = {}
        for i in range(max_tokens):
            try:
                token_text = self.tokenizer.decode([i], skip_special_tokens=False)
                self.token_id_to_text[i] = token_text
            except:
                self.token_id_to_text[i] = f"<UNK_{i}>"
                
        print(f"✅ Создана матрица embedding'ов для {max_tokens} токенов (float32)")
    
    def _compress_audio_features(self, audio_features, compression_rate_k):
        """Сжимает аудио-признаки по K-фактору"""
        batch_size, seq_len, hidden_dim = audio_features.shape
        
        new_seq_len = (seq_len // compression_rate_k) * compression_rate_k
        audio_features = audio_features[:, :new_seq_len, :]
        
        reshaped = audio_features.view(batch_size, new_seq_len // compression_rate_k, compression_rate_k, hidden_dim)
        compressed = reshaped.view(batch_size, new_seq_len // compression_rate_k, compression_rate_k * hidden_dim)
        
        return compressed
    
    def find_nearest_tokens(self, projected_audio_embeds: torch.Tensor, top_k: int = 10) -> List[Dict]:
        """
        Находит ближайшие токены для каждого audio embedding'а
        🔧 ИСПРАВЛЕНО: Все операции в float32
        """
        with torch.no_grad():
            # 🔧 ИСПРАВЛЕНИЕ: Конвертируем в float32
            audio_embeds_f32 = projected_audio_embeds.float()
            
            # Нормализуем embedding'ы для косинусной близости
            audio_norm = F.normalize(audio_embeds_f32, p=2, dim=-1)
            vocab_norm = F.normalize(self.vocab_embeddings, p=2, dim=-1)
            
            # Вычисляем косинусную близость
            similarity_matrix = torch.mm(audio_norm, vocab_norm.t())
            
            # Находим top_k для каждого временного шага
            top_similarities, top_indices = torch.topk(similarity_matrix, top_k, dim=-1)
            
            results = []
            for t in range(projected_audio_embeds.size(0)):
                step_result = {
                    'timestep': t,
                    'nearest_tokens': [],
                    'similarities': top_similarities[t].cpu().numpy().tolist(),
                    'token_ids': top_indices[t].cpu().numpy().tolist()
                }
                
                for i in range(top_k):
                    token_id = top_indices[t, i].item()
                    similarity = top_similarities[t, i].item()
                    token_text = self.token_id_to_text.get(token_id, f"<UNK_{token_id}>")
                    
                    step_result['nearest_tokens'].append({
                        'token_id': token_id,
                        'token_text': token_text,
                        'similarity': similarity
                    })
                
                results.append(step_result)
                
        return results
    
    def create_tsne_visualization(self, 
                                  projected_audio_list: List[torch.Tensor], 
                                  target_embeds_list: List[torch.Tensor],
                                  labels: List[str] = None,
                                  step: int = 0):
        """
        Создает t-SNE визуализацию
        🔧 ИСПРАВЛЕНО: Все операции в float32
        """
        print("📊 Создание t-SNE визуализации...")
        
        # Объединяем все embedding'ы
        all_embeddings = []
        all_types = []
        all_labels = []
        
        for i, (audio_emb, text_emb) in enumerate(zip(projected_audio_list, target_embeds_list)):
            # 🔧 ИСПРАВЛЕНИЕ: Конвертируем в float32 перед усреднением
            audio_mean = audio_emb.float().mean(dim=0).detach().cpu().numpy()
            text_mean = text_emb.float().mean(dim=0).detach().cpu().numpy()
            
            all_embeddings.extend([audio_mean, text_mean])
            all_types.extend(['Audio', 'Text'])
            
            label = labels[i] if labels else f"Sample_{i}"
            all_labels.extend([f"Audio_{label}", f"Text_{label}"])
        
        # Применяем t-SNE
        embeddings_array = np.array(all_embeddings)
        
        # PCA для уменьшения размерности, если это возможно и необходимо
        n_samples, n_features = embeddings_array.shape
        if n_features > 50:
            # ❗️ ИСПРАВЛЕНИЕ: n_components не может быть больше, чем количество образцов
            pca_n_components = min(50, n_samples)
            if pca_n_components < n_features:
                print(f"📉 PCA: уменьшение размерности с {n_features} до {pca_n_components}")
                pca = PCA(n_components=pca_n_components)
                embeddings_array = pca.fit_transform(embeddings_array)
            
        # ❗️ ИСПРАВЛЕНИЕ: perplexity должен быть меньше n_samples
        perplexity_value = max(1, min(30, n_samples - 1))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
        embeddings_2d = tsne.fit_transform(embeddings_array)
        
        # Создаем DataFrame
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1], 
            'type': all_types,
            'label': all_labels
        })
        
        # Создаем визуализацию
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x='x', y='y', hue='type', style='type', s=100, alpha=0.7)
        
        plt.title(f"t-SNE: Audio vs Text Embeddings (Step {step})", fontsize=16)
        plt.xlabel("t-SNE Component 1", fontsize=12)
        plt.ylabel("t-SNE Component 2", fontsize=12)
        plt.legend(title="Embedding Type", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_to_wandb:
            wandb.log({f"analysis/tsne_step_{step}": wandb.Image(plt)})
            print("✅ t-SNE визуализация отправлена в W&B")
        else:
            plt.show()
            print("✅ t-SNE визуализация отображена локально")
        
        plt.close()
        return df
    
    def create_similarity_heatmap(self, 
                                  projected_audio: torch.Tensor, 
                                  target_embeds: torch.Tensor,
                                  title: str = "Audio-Text Cosine Similarity",
                                  step: int = 0):
        """
        Создает матрицу косинусной близости
        🔧 ИСПРАВЛЕНО: Все операции в float32
        """
        print("🔥 Создание матрицы косинусной близости...")
        
        with torch.no_grad():
            # 🔧 ИСПРАВЛЕНИЕ: Конвертируем в float32
            audio_f32 = projected_audio.float()
            target_f32 = target_embeds.float()
            
            # Нормализуем для косинусной близости
            audio_norm = F.normalize(audio_f32, p=2, dim=-1)
            text_norm = F.normalize(target_f32, p=2, dim=-1)
            
            # Вычисляем попарную косинусную близость
            similarity_matrix = torch.mm(audio_norm, text_norm.t())
            similarity_np = similarity_matrix.detach().cpu().numpy()
        
        # Создаем тепловую карту
        plt.figure(figsize=(12, 8))
        
        sns.heatmap(similarity_np, 
                   cmap='viridis',
                   cbar=True,
                   xticklabels=False,
                   yticklabels=False,
                   cbar_kws={'label': 'Cosine Similarity'})
        
        plt.title(f"{title} (Step {step})", fontsize=16)
        plt.xlabel('Text Tokens', fontsize=12)
        plt.ylabel('Audio Tokens', fontsize=12)
        
        plt.figtext(0.02, 0.02, 
                   f'Min: {similarity_np.min():.3f}, Max: {similarity_np.max():.3f}, Mean: {similarity_np.mean():.3f}', 
                   fontsize=10, ha='left')
        
        plt.tight_layout()
        
        if self.save_to_wandb:
            # wandb.log({
            #     f"analysis/similarity_heatmap_step_{step}": wandb.Image(plt),
            #     f"analysis/similarity_min_step_{step}": float(similarity_np.min()),
            #     f"analysis/similarity_max_step_{step}": float(similarity_np.max()),
            #     f"analysis/similarity_mean_step_{step}": float(similarity_np.mean()),
            #     f"analysis/similarity_std_step_{step}": float(similarity_np.std())
            # })
            print("✅ Матрица косинусной близости создана (логирование в W&B отключено)")
        else:
            plt.show()
            print("✅ Матрица косинусной близости отображена локально")
        
        plt.close()
        
        return {
            'similarity_matrix': similarity_np,
            'min_similarity': float(similarity_np.min()),
            'max_similarity': float(similarity_np.max()),
            'mean_similarity': float(similarity_np.mean()),
            'std_similarity': float(similarity_np.std())
        }
    
    def create_nearest_tokens_string(self, projected_audio_embeds: torch.Tensor, max_length: int = 50) -> Dict:
        """
        Создает строку из ближайших токенов
        🔧 ИСПРАВЛЕНО: Все операции в float32
        """
        print("🎯 Создание строки ближайших токенов...")
        
        with torch.no_grad():
            # 🔧 ИСПРАВЛЕНИЕ: Конвертируем в float32
            audio_f32 = projected_audio_embeds.float()
            
            # Нормализуем embedding'ы
            audio_norm = F.normalize(audio_f32, p=2, dim=-1)
            vocab_norm = F.normalize(self.vocab_embeddings, p=2, dim=-1)
            
            # Вычисляем косинусную близость
            similarity_matrix = torch.mm(audio_norm, vocab_norm.t())
            
            # Находим самый близкий токен для каждого временного шага
            top_similarities, top_indices = torch.topk(similarity_matrix, 1, dim=-1)
            
            # Создаем строку из ближайших токенов
            nearest_tokens_list = []
            similarities_list = []
            
            seq_len = min(projected_audio_embeds.size(0), max_length)
            
            for t in range(seq_len):
                token_id = top_indices[t, 0].item()
                similarity = top_similarities[t, 0].item()
                token_text = self.token_id_to_text.get(token_id, f"<UNK_{token_id}>")
                
                nearest_tokens_list.append(token_text)
                similarities_list.append(similarity)
            
            # Создаем результирующую строку
            nearest_tokens_string = "".join(nearest_tokens_list)
            readable_string = " ".join([token.strip() for token in nearest_tokens_list if token.strip()])
            
            # Статистики
            avg_similarity = sum(similarities_list) / len(similarities_list)
            
            result = {
                'nearest_tokens_raw': nearest_tokens_string,
                'nearest_tokens_readable': readable_string,
                'individual_tokens': nearest_tokens_list,
                'similarities': similarities_list,
                'statistics': {
                    'avg_similarity': avg_similarity,
                    'min_similarity': min(similarities_list),
                    'max_similarity': max(similarities_list),
                    'sequence_length': seq_len
                },
                'token_details': [
                    {
                        'timestep': t,
                        'token': nearest_tokens_list[t],
                        'similarity': similarities_list[t]
                    }
                    for t in range(seq_len)
                ]
            }
            
        return result
    
    def run_comprehensive_analysis(self, 
                                   batch: Dict,
                                   compression_rate_k: int,
                                   prefix_embeds: torch.Tensor,
                                   current_step: int = 0,
                                   perform_tsne: bool = True) -> Dict:
        """
        Запускает комплексный анализ для одного батча
        🔧 ИСПРАВЛЕНО: Все операции в float32
        """
        print(f"🔬 Запуск комплексного анализа для шага {current_step}...")
        
        # Переводим модели в eval режим
        self.projector.eval()
        self.wav2vec2.eval()
        self.model.eval()
        
        with torch.no_grad():
            # Получаем embedding'ы
            input_values = batch["input_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            
            # Аудио pipeline
            audio_embeds = self.wav2vec2(input_values.to(torch.bfloat16)).last_hidden_state
            compressed_audio = self._compress_audio_features(audio_embeds, compression_rate_k)
            projected_audio = self.projector(compressed_audio.float()).float()  # ← float32
            
            # Текстовые embedding'ы
            # ❗️ ИСПРАВЛЕНИЕ: Заменяем -100 (используется для лосса) на pad_token_id для embedding'а
            input_ids_for_embedding = input_ids.clone()
            input_ids_for_embedding[input_ids_for_embedding == -100] = self.tokenizer.pad_token_id
            target_embeds = self.model.get_input_embeddings()(input_ids_for_embedding).float()  # ← float32
            
            results = {
                # Возвращаем эмбеддинги, чтобы их можно было собрать снаружи
                'projected_audio': projected_audio,
                'target_embeds': target_embeds
            }
            
            # Анализируем первый пример в батче
            first_projected = projected_audio[0]  # [seq_len, hidden_dim]
            first_target = target_embeds[0]       # [seq_len, hidden_dim]
            
            # 1. Создаем строку ближайших токенов
            print("🎯 Анализ ближайших токенов...")
            tokens_string_result = self.create_nearest_tokens_string(first_projected)
            results['tokens_string'] = tokens_string_result
            
            # Выводим результат
            print(f"\n📝 АУДИО → MLP → ТОКЕНЫ:")
            print(f"   Читаемая версия: '{tokens_string_result['nearest_tokens_readable']}'")
            print(f"   Средняя similarity: {tokens_string_result['statistics']['avg_similarity']:.3f}")
            
            # 2. Создаем матрицу косинусной близости
            similarity_stats = self.create_similarity_heatmap(
                first_projected, 
                first_target,
                title=f"Audio-Text Similarity Step {current_step}",
                step=current_step
            )
            results['similarity_stats'] = similarity_stats
            
            # 3. t-SNE визуализация (если флаг включен и больше одного примера)
            if perform_tsne and projected_audio.size(0) > 1:
                print("📈 Создание t-SNE визуализации...")
                projected_list = [projected_audio[i] for i in range(min(3, projected_audio.size(0)))]
                target_list = [target_embeds[i] for i in range(min(3, target_embeds.size(0)))]
                labels = [f"Sample_{i}" for i in range(len(projected_list))]
                
                tsne_df = self.create_tsne_visualization(
                    projected_list,
                    target_list, 
                    labels=labels,
                    step=current_step
                )
                results['tsne_data'] = tsne_df.to_dict()
            
            print(f"✅ Комплексный анализ завершен для шага {current_step}")
            return results

def load_model_for_analysis(checkpoint_path, device="cuda"):
    """
    Загружает обученную модель для анализа
    Воспроизводит полную инициализацию из audio_projector_training_lora.ipynb
    """
    print(f"📂 Загрузка модели из {checkpoint_path}...")
    
    # === 1. Параметры конфигурации (из основного файла) ===
    model_id = "google/gemma-3-4b-pt"
    audio_model_name = "facebook/wav2vec2-xls-r-300m"
    input_dim = 1024
    output_dim = 2560
    compression_rate_k = 3
    
    # === 2. Инициализация токенайзера ===
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # === 3. Квантизация ===
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    # === 4. Инициализация Gemma с LoRA ===
    hf_token = os.getenv('HF_TOKEN')
    multi_cfg = AutoConfig.from_pretrained(model_id, token=hf_token)
    
    text_cfg_dict = multi_cfg.text_config.to_dict()
    text_cfg_dict["vocab_size"] = 262208
    text_cfg_dict.update({
        "bos_token_id": tokenizer.bos_token_id, 
        "eos_token_id": tokenizer.eos_token_id, 
        "pad_token_id": tokenizer.pad_token_id
    })
    
    text_cfg = Gemma3TextConfig(**text_cfg_dict)
    gemma_model = Gemma3ForCausalLM.from_pretrained(
        model_id,
        config=text_cfg,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="cuda",
        token=hf_token
    )
    
    gemma_model.gradient_checkpointing_enable()
    gemma_model = prepare_model_for_kbit_training(gemma_model)
    
    # LoRA конфигурация
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
        lora_dropout=0.2,
        bias="none",
        init_lora_weights=False,
        task_type="CAUSAL_LM"
    )
    
    gemma_model = get_peft_model(gemma_model, lora_config)
    
    # === 5. Инициализация Wav2Vec2 ===
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(audio_model_name)
    wav2vec2 = Wav2Vec2Model.from_pretrained(audio_model_name)
    wav2vec2 = wav2vec2.to(torch.bfloat16).to(device)
    wav2vec2.eval()
    for param in wav2vec2.parameters():
        param.requires_grad = False
    
    # === 6. Инициализация AudioProjector ===
    class AudioProjector(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048):
            super().__init__()

            self.proj = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
            
            for layer in self.proj:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
        def forward(self, x):
            # Always return in bfloat16 for consistency with model
            return self.proj(x.float()).to(torch.bfloat16)
        
        def get_l2_norm(self):
            total_norm = 0.0
            for param in self.parameters():
                total_norm += param.data.norm(2).item() ** 2
            return total_norm ** 0.5
    
    projector_input_dim = input_dim * compression_rate_k
    projector_hidden_dim = 2048
    projector = AudioProjector(projector_input_dim, output_dim, hidden_dim=projector_hidden_dim).to(device).float()
    
    # === 7. Загрузка весов из чекпоинта ===
    print(f"🔄 Загрузка весов из {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Загружаем веса MLP проектора
    if 'projector_state_dict' in checkpoint:
        projector.load_state_dict(checkpoint['projector_state_dict'])
        print("✅ Веса MLP проектора загружены")
    else:
        print("⚠️ Веса MLP проектора не найдены в чекпоинте")
    
    # Загружаем веса LoRA
    if 'lora_state_dict' in checkpoint:
        # Загружаем только LoRA веса, игнорируем остальные
        lora_state_dict = checkpoint['lora_state_dict']
        missing_keys, unexpected_keys = gemma_model.load_state_dict(lora_state_dict, strict=False)
        if len(missing_keys) == 0:
            print("✅ Веса LoRA загружены")
        else:
            print(f"⚠️ Некоторые LoRA веса не найдены: {len(missing_keys)} ключей")
    else:
        print("⚠️ Веса LoRA не найдены в чекпоинте")
    
    # Информация о чекпоинте
    if 'epoch' in checkpoint and 'step' in checkpoint:
        print(f"📊 Чекпоинт: эпоха {checkpoint['epoch']}, шаг {checkpoint['step']}")
    
    if 'loss' in checkpoint:
        print(f"📊 Loss: {checkpoint['loss']:.4f}")
    
    print("✅ Модель полностью загружена для анализа")
    
    return projector, gemma_model, wav2vec2, tokenizer

def run_analysis_on_checkpoint(checkpoint_path, data_path="transcripts.jsonl", zip_path="LibriSpeech.zip", device="cuda", num_samples=5):
    """
    Запускает анализ на сохраненной модели с данными LibriSpeech
    
    Args:
        checkpoint_path: путь к чекпоинту модели
        data_path: путь к JSONL файлу с транскрипциями
        zip_path: путь к ZIP архиву с аудиофайлами LibriSpeech
        device: устройство для вычислений
        num_samples: количество образцов для анализа
    """
    print(f"🔬 Запуск анализа модели: {checkpoint_path}")
    print(f"📊 Данные: {data_path} + {zip_path}")
    
    # Инициализируем W&B для анализа
    wandb.init(
        project="audio-projector-analysis",
        name=f"analysis_{os.path.basename(checkpoint_path)}",
        tags=["analysis", "standalone"],
        config={
            "checkpoint_path": checkpoint_path,
            "data_path": data_path,
            "zip_path": zip_path,
            "num_samples": num_samples
        }
    )
    
    try:
        # === 1. Загружаем модель ===
        projector, gemma_model, wav2vec2, tokenizer = load_model_for_analysis(checkpoint_path, device)
        
        # === 2. Создаем анализатор ===
        analyzer = AudioTextEmbeddingAnalyzer(
            model=gemma_model,
            projector=projector,
            wav2vec2=wav2vec2,
            tokenizer=tokenizer,
            device=device,
            save_to_wandb=True
        )
        
        # === 3. Загружаем данные LibriSpeech ===
        print(f"📂 Загрузка данных из {data_path}...")
        
        # Загружаем JSONL файл с транскрипциями
        with open(data_path, "r", encoding="utf-8") as f:
            all_data = [json.loads(line) for line in f]
        
        print(f"📊 Загружено {len(all_data)} записей из {data_path}")
        
        # Берем случайную выборку для анализа
        import random
        random.seed(42)  # Для воспроизводимости
        sample_data = random.sample(all_data, min(num_samples, len(all_data)))
        
        print(f"🎯 Выбрано {len(sample_data)} образцов для анализа")
        
        # === 4. Создаем датасет и DataLoader ===
        from torch.utils.data import Dataset, DataLoader
        from transformers import Wav2Vec2FeatureExtractor
        import torchaudio
        import zipfile
        import io
        
        # Простая версия AudioTextDataset для анализа
        class AnalysisDataset(Dataset):
            def __init__(self, data, tokenizer, feature_extractor, zip_path):
                self.data = data
                self.tokenizer = tokenizer
                self.feature_extractor = feature_extractor
                
                # Открываем ZIP файл
                try:
                    self.zip_file = zipfile.ZipFile(zip_path, 'r')
                    # Создаем манифест файлов
                    self.zip_manifest = {
                        os.path.basename(p): p 
                        for p in self.zip_file.namelist() 
                        if p.lower().endswith(('.flac', '.wav', '.mp3'))
                    }
                    print(f"✅ ZIP архив открыт: {len(self.zip_manifest)} аудиофайлов")
                except Exception as e:
                    print(f"⚠️ Ошибка открытия ZIP: {e}")
                    self.zip_file = None
                    self.zip_manifest = {}
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                audio_path = item.get("audio_filepath", item.get("audio_path", ""))
                text = item.get("text", item.get("speaker_text", ""))
                
                # Загружаем аудио из ZIP
                try:
                    if self.zip_file and self.zip_manifest:
                        filename = os.path.basename(audio_path)
                        found_path = self.zip_manifest.get(filename)
                        if found_path:
                            with self.zip_file.open(found_path) as audio_file:
                                audio_data = audio_file.read()
                                waveform, sr = torchaudio.load(io.BytesIO(audio_data))
                        else:
                            raise FileNotFoundError(f"Файл {filename} не найден в ZIP")
                    else:
                        waveform, sr = torchaudio.load(audio_path)
                        
                except Exception as e:
                    print(f"⚠️ Ошибка загрузки {audio_path}: {e}")
                    # Создаем пустой аудио
                    waveform = torch.zeros(1, 16000)
                    sr = 16000
                
                # Ресемплинг если нужен
                if sr != self.feature_extractor.sampling_rate:
                    waveform = torchaudio.functional.resample(waveform, sr, self.feature_extractor.sampling_rate)
                
                # Конвертируем в моно
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                # Z-нормализация
                waveform_np = waveform.squeeze().numpy()
                waveform_mean = np.mean(waveform_np)
                waveform_std = np.std(waveform_np)
                if waveform_std > 1e-8:
                    waveform_np = (waveform_np - waveform_mean) / waveform_std
                
                # Обработка аудио
                audio_inputs = self.feature_extractor(
                    waveform_np,
                    sampling_rate=self.feature_extractor.sampling_rate,
                    return_tensors="pt"
                )
                
                # Обработка текста
                text_inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                return {
                    "input_values": audio_inputs.input_values.squeeze(0),
                    "input_ids": text_inputs.input_ids.squeeze(0),
                    "attention_mask": text_inputs.attention_mask.squeeze(0),
                    "text": text,
                    "audio_path": audio_path
                }
        
        # Создаем feature extractor
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
        
        # Создаем датасет и загрузчик
        analysis_dataset = AnalysisDataset(sample_data, tokenizer, feature_extractor, zip_path)
        
        def collate_fn(batch):
            from torch.nn.utils.rnn import pad_sequence
            input_values = [item['input_values'] for item in batch]
            input_ids = [item['input_ids'] for item in batch]
            attention_mask = [item['attention_mask'] for item in batch]
            
            input_values = pad_sequence(input_values, batch_first=True)
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
            attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
            
            return {
                'input_values': input_values,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'texts': [item['text'] for item in batch],
                'audio_paths': [item['audio_path'] for item in batch]
            }
        
        analysis_loader = DataLoader(
            analysis_dataset,
            batch_size=min(2, len(sample_data)),  # Небольшой batch для анализа
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # === 5. Создаем prefix embeddings ===
        prefix = "Transcribe speech to text."
        prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            prefix_embeds = gemma_model.get_input_embeddings()(prefix_ids).to(dtype=torch.bfloat16)
        
        # === 6. Запускаем анализ ===
        print(f"\n🔬 === ЗАПУСК АНАЛИЗА EMBEDDING'ОВ ===")
        
        compression_rate_k = 3
        step_counter = 0
        
        # Списки для сбора эмбеддингов для итогового t-SNE
        collected_projected_embeds = []
        collected_target_embeds = []
        collected_labels = []

        for batch_idx, batch in enumerate(analysis_loader):
            print(f"\n📊 Анализ батча {batch_idx + 1}/{len(analysis_loader)}")
            print(f"   Аудио файлы: {batch['audio_paths']}")
            print(f"   Тексты: {[text[:50] + '...' if len(text) > 50 else text for text in batch['texts']]}")
            
            # Запускаем комплексный анализ
            try:
                # Запускаем анализ для каждого батча, но отключаем t-SNE внутри
                results = analyzer.run_comprehensive_analysis(
                    batch=batch,
                    compression_rate_k=compression_rate_k,
                    prefix_embeds=prefix_embeds,
                    current_step=step_counter,
                    perform_tsne=False
                )

                # Собираем эмбеддинги для итогового t-SNE
                projected_audio = results['projected_audio']
                target_embeds = results['target_embeds']
                for i in range(projected_audio.size(0)):
                    collected_projected_embeds.append(projected_audio[i])
                    collected_target_embeds.append(target_embeds[i])
                    collected_labels.append(f"Sample_{step_counter + i}")
                
                # Логируем результаты в W&B (кроме t-SNE)
                # wandb.log({
                #     f"analysis/batch_{batch_idx}/tokens_string": results.get('tokens_string', {}).get('nearest_tokens_readable', ''),
                #     f"analysis/batch_{batch_idx}/avg_similarity": results.get('tokens_string', {}).get('statistics', {}).get('avg_similarity', 0),
                #     f"analysis/batch_{batch_idx}/similarity_mean": results.get('similarity_stats', {}).get('mean_similarity', 0),
                #     f"analysis/batch_{batch_idx}/similarity_std": results.get('similarity_stats', {}).get('std_similarity', 0),
                #     "step": step_counter
                # })
                
                print(f"✅ Анализ батча {batch_idx + 1} завершен")
                
            except Exception as e:
                print(f"❌ Ошибка анализа батча {batch_idx + 1}: {e}")
                continue
            
            step_counter += len(batch['texts'])

        # После цикла создаем одну большую t-SNE визуализацию
        if collected_projected_embeds:
            print(f"\n📈 Создание итоговой t-SNE визуализации для {len(collected_projected_embeds)} образцов...")
            analyzer.create_tsne_visualization(
                projected_audio_list=collected_projected_embeds,
                target_embeds_list=collected_target_embeds,
                labels=collected_labels,
                step=9999  # Уникальный шаг для итогового графика
            )
        
        print(f"\n🎉 === АНАЛИЗ ЗАВЕРШЕН ===")
        print(f"📊 Проанализировано батчей: {step_counter}")
        print(f"🔗 Результаты сохранены в W&B: {wandb.run.url}")
        
    except Exception as e:
        print(f"❌ Критическая ошибка анализа: {e}")
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    # Пример использования
    checkpoint_path = "/home/jovyan/persistent_volume/checkpoint_bs4_step_12000.pt"
    data_path = "transcripts.jsonl"  # Ваш JSONL файл с LibriSpeech
    zip_path = "LibriSpeech.zip"    # Ваш ZIP с аудиофайлами
    
    run_analysis_on_checkpoint(
        checkpoint_path=checkpoint_path,
        data_path=data_path,
        zip_path=zip_path,
        num_samples=50  # Анализируем 50 случайных образцов для 100 точек на t-SNE
    ) 
