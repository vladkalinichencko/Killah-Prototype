
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer, Wav2Vec2FeatureExtractor, GemmaForCausalLM, GemmaConfig, QuantoConfig

@dataclass
class TrainingConfig:
    GEMMA_MODEL_ID: str = "google/gemma-3-4b-pt"
    XLSR_MODEL_ID: str = "facebook/wav2vec2-xls-r-300m"
    EPOCHS = 3
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class AudioProjector(nn.Module):
    def __init__(self, audio_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        self.layer1 = nn.Linear(audio_hidden_size, llm_hidden_size * 2)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(llm_hidden_size * 2, llm_hidden_size)

    def forward(self, audio_embeds: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.gelu(self.layer1(audio_embeds)))

def create_gemma_config(vocab_size, pad_token_id):
    return GemmaConfig(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        hidden_size=2560,
        intermediate_size=10240,
        num_hidden_layers=34,
        num_attention_heads=20,
        num_key_value_heads=20,
        head_dim=128,
        model_type="gemma"
    )

class AudioGemmaModel(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.GEMMA_MODEL_ID)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        gemma_config = create_gemma_config(self.tokenizer.vocab_size, self.tokenizer.pad_token_id)
        
        self.gemma = GemmaForCausalLM.from_pretrained(
            config.GEMMA_MODEL_ID, 
            config=gemma_config,
            quantization_config=QuantoConfig(weights="int4"),
            device_map={"": config.DEVICE},
            torch_dtype=torch.bfloat16
        )
        self.gemma.resize_token_embeddings(len(self.tokenizer))
        
        self.audio_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.XLSR_MODEL_ID)
        self.audio_encoder = AutoModel.from_pretrained(config.XLSR_MODEL_ID).to(config.DEVICE)
        self.projector = AudioProjector(self.audio_encoder.config.hidden_size, self.gemma.config.hidden_size).to(config.DEVICE)
        
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        for param in self.gemma.parameters():
            param.requires_grad = False

def forward(self, audio_values, input_ids, attention_mask):
    audio_embeds = self.audio_encoder(audio_values).last_hidden_state
    projected_audio = self.projector(audio_embeds)
    text_embeds = self.gemma.get_input_embeddings()(input_ids)
    
    combined_embeds = torch.cat([projected_audio, text_embeds], dim=1)
    combined_embeds = combined_embeds.to(self.gemma.device).to(self.gemma.dtype)
    audio_mask = torch.ones(projected_audio.shape[:2], dtype=torch.long, device=projected_audio.device)
    combined_mask = torch.cat([audio_mask, attention_mask], dim=1)
    
    return self.gemma(inputs_embeds=combined_embeds, attention_mask=combined_mask).logits

AudioGemmaModel.forward = forward
config = TrainingConfig()
model = AudioGemmaModel(config)
model.eval()


dummy_audio = [np.random.randn(32000).astype(np.float32) for _ in range(config.BATCH_SIZE)]
audio_processed = model.audio_extractor(dummy_audio, return_tensors="pt", sampling_rate=16000, padding=True)
audio_values = audio_processed.input_values.to(config.DEVICE)
dummy_texts = ["Test text"] * config.BATCH_SIZE
text_processed = model.tokenizer(dummy_texts, return_tensors="pt", padding=True, max_length=32)
input_ids = text_processed.input_ids.to(config.DEVICE)
attention_mask = text_processed.attention_mask.to(config.DEVICE)

print(f"Audio shape: {audio_values.shape}")
print(f"Text shape: {input_ids.shape}")

print(f"Используемое устройство: {config.DEVICE}")

raw_audio_sr = 16000
dummy_audio_waveforms = [np.random.randn(raw_audio_sr * 2).astype(np.float32) for _ in range(config.BATCH_SIZE)]
audio_processed = model.audio_extractor(dummy_audio_waveforms, return_tensors="pt", sampling_rate=raw_audio_sr, padding=True)
audio_input_values = audio_processed.input_values.to(config.DEVICE)
print(f"Форма audio_input_values: {audio_input_values.shape}, устройство: {audio_input_values.device}")

dummy_texts = ["Это пример текста для модели Gemma." for _ in range(config.BATCH_SIZE)]
text_tokenized = model.tokenizer(dummy_texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
input_ids = text_tokenized.input_ids.to(config.DEVICE)
attention_mask = text_tokenized.attention_mask.to(config.DEVICE)
print(f"Форма input_ids: {input_ids.shape}, устройство: {input_ids.device}")
print(f"Форма attention_mask: {attention_mask.shape}, устройство: {attention_mask.device}")

print("\nВыполнение тестового прогона модели (forward pass)...")
try:
    with torch.no_grad():
        logits = model(audio_input_values, input_ids, attention_mask)
    print(f"Success! Logits shape: {logits.shape}")
except Exception as e:
    print(f"КРИТИЧЕСКАЯ ОШИБКА во время forward pass: {e}")
    import traceback
    traceback.print_exc()
print("\n--- Тестовый запуск завершён ---")


# Sampling from logits to generate varied outputs
import torch.nn.functional as F
batch_size, seq_len, vocab_size = logits.shape
sampled_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=logits.device)
for t in range(seq_len):
    probs_t = F.softmax(logits[:, t, :], dim=-1)
    sampled_ids[:, t] = torch.multinomial(probs_t, num_samples=1).squeeze(-1)
sampled_texts = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in sampled_ids]
print('Sampled texts:', sampled_texts)
