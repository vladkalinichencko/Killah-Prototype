# Killah Language Model Implementation Guide

**Last Updated:** June 4, 2025

## 1. Project Overview

### 1.1. Project Name

lil Pushkin

### 1.2. Core Goal

Develop a persona-conditioned language model for writing assistance through deep personalization. The model uses the Google Gemma 3 4B base model, which has not undergone extensive Reinforcement Learning from Human Feedback (RLHF). This choice aims to preserve the model's raw stylistic capabilities, making it adaptable for genuine personalization and less prone to generic outputs sometimes associated with heavily aligned models. By using a non-RLHF base model, "lil Pushkin" seeks to maintain flexibility and creativity for tasks like story generation, rephrasing, and continuation.

The system incorporates native multimodal capabilities, seamlessly handling both text and audio input through a unified persona-conditioned architecture. Users can dictate content, issue voice commands, and receive audio feedback, all while maintaining their personal writing style and voice characteristics. The audio processing is deeply integrated with the text understanding, allowing for natural transitions between typing and speaking during the writing process.

All capabilities are built upon this base using a LoRA-centric approach, starting with a foundational PersonaPlugs conditioning layer, upon which all task-specific LoRA adapters are built. This ensures every capability inherently reflects the user's writing style and voice.

### 1.3. Architecture Principles

The architecture of "lil Pushkin" is built around **MLP-Projector Integration** - a multimodal approach inspired by Llama-AVSR research where audio embeddings are projected into LLM text space through learnable adapters. This follows the proven methodology from "Large Language Models are Strong Audio-Visual Speech Recognition Learners" (ICASSP 2025).

We start with a **Clean Base Model** using Google Gemma 3 4B that hasn't undergone extensive RLHF (Reinforcement Learning from Human Feedback). Most commercial models are heavily fine-tuned to be "safe" and "helpful," which often makes them sound generic and corporate. By starting with a less processed model, we maintain the raw stylistic capabilities that make genuine personalization possible.

Our **Multimodal Architecture** consists of three core components working in harmony:

**Audio Processing Pipeline** uses a frozen Wav2Vec2-XLS-R-300M encoder to extract audio embeddings, which are then projected into Gemma's text space through a specialized MLP projector. This projector consists of LayerNorm → Linear(audio_dim, llm_dim×2) → GELU → Dropout(0.1) → Linear(llm_dim×2, llm_dim) → LayerNorm architecture for stable training.

**Persona LoRA Layer** is trained on top of the base model to learn stylistic adaptation using the PersonaPlugs methodology. This adapter focuses purely on capturing and reproducing writing style patterns while working seamlessly with audio inputs.

**Task-Specific LoRAs** are built upon the Persona LoRA foundation for specific writing tasks (continuation, rephrasing, summarization). These LoRAs inherit the personalization capabilities while adding task-specific behaviors.

**Training Pipeline** follows a structured approach:

- **Audio MLP Projector Training:** Curriculum learning from basic transcription to emotion recognition
- **PersonaPlugs MLP Projector Training:** Document embeddings to LLM space projection
- **Task-Specific LoRA Training:** Specialized adapters for different writing tasks
- **DPO Refinement:** Quality optimization across all components

**Dynamic Composition at Runtime** follows a consistent pattern where both audio and persona understanding are available:

- **Pure autocomplete only:** `Base + Task` (minimal composition for maximum speed)
- **Audio transcription:** `Base + Audio-Projector` (direct speech-to-text)
- **Personalized writing:** `Base + Persona-Projector + Persona + Task` (full personalization)
- **Multimodal personalized:** `Base + Audio-Projector + Persona-Projector + Persona + Task` (complete system)

This architecture ensures that **both MLP projectors work together seamlessly**, providing consistent multimodal and personalized experience while maintaining computational efficiency through selective quantization.

Finally, **Mixed-Precision Training Strategy** optimizes memory usage: Gemma (INT4 weights + BF16 activations, frozen), Audio Encoder (BF16, frozen), MLP Projector (FP32, trainable), ensuring training stability while maximizing efficiency.

### 1.4. Implementation Timeline

**Duration:** Adjusted for rapid development (target ~2-3 weeks intensive work)

### 1.4. Implementation Timeline

**Duration:** Adjusted for rapid development (target ~2-3 weeks intensive work)

#### Week 1: Audio MLP Projector Development & Curriculum Learning

The first week focuses on implementing and training the core audio processing pipeline using curriculum learning methodology inspired by the Llama-AVSR architecture.

**MLP Projector Architecture Setup (Days 1-2):**
We begin with environment setup and acquiring the Gemma 3 4B base model and Wav2Vec2-XLS-R-300M audio encoder. The MLP projector implements a two-layer architecture with LayerNorm for stability:

**Curriculum Learning Stage 1: Basic Transcription (Days 2-4):**
Starting with simple audio-to-text transcription using LibriSpeech and Common Voice datasets. Mixed-precision training strategy: Gemma (INT4 weights + BF16 activations, frozen), Audio Encoder (BF16, frozen), MLP Projector (FP32, trainable). Target metrics: Word Error Rate (WER) < 5%.

**Curriculum Learning Stage 2: Emotion Recognition (Days 4-6):**
Expanding projector capabilities to recognize emotional content in speech using labeled emotional speech datasets (RAVDESS, SAVEE, Russian emotional speech corpora). Additional metrics: Emotion Recognition Accuracy, emotional consistency scores.

**Target Outcomes:** Trained MLP projector with transcription and emotion recognition capabilities
**GPU Requirements:** 40-60 A100 GPU hours total (curriculum learning requires more data)

#### Week 2: Persona LoRA Integration & Task-Specific Training

Week 2 focuses on integrating persona adaptation with the trained audio projector and developing task-specific capabilities.

**Persona LoRA Development (Days 1-3):**
Training persona adaptation LoRA on diverse author corpora from Project Gutenberg and other sources. This LoRA works on top of the base model and coordinates with the audio projector to provide personalized multimodal responses. Rank 64, alpha 128 parameters for style adaptation.

**Task-Specific LoRA Training (Days 3-5):**
Building specialized LoRAs for writing tasks (continuation, rephrasing) that can work with both audio input and persona conditioning:

- `Base + Audio-Projector + Persona + Task_Continue` for voice-guided personalized writing
- `Base + Audio-Projector + Task_Transcribe` for direct speech-to-text
- `Base + Persona + Task_Continue` for text-only personalized writing

**WQRM Development (Days 5-7):**
Implement and validate our Writing Quality Reward Model using edit preference pairs and quality gradient examples, specifically adapted for multimodal inputs.

**Target Outcomes:** Integrated persona and task-specific capabilities working with audio
**GPU Requirements:** 35-50 A100 GPU hours for LoRA training

#### Week 3: DPO Refinement & Deployment Optimization

Week 3 implements DPO refinement and prepares the optimized system for deployment.

**DPO Refinement with Multimodal Data (Days 1-4):**
Apply Direct Preference Optimization to the integrated audio-persona-task compositions using our WQRM. Focus on improving quality while maintaining audio understanding and personalization capabilities.

**Quantization & Deployment Optimization (Days 4-7):**
Implement production-ready quantization strategy:
- Gemma: INT4 weights + BF16 activations (70% memory reduction)
- Audio Encoder: BF16 (50% memory reduction)  
- MLP Projector: FP32 (training stability)
- ExecuTorch preparation for optimal inference

**Performance Metrics & Validation:**
- **Audio Quality:** Word Error Rate (WER), Real-Time Factor (RTF)
- **Emotion Recognition:** Accuracy, F1-score per emotion class
- **Text Quality:** BLEU/ROUGE scores, WQRM composite scores
- **Personalization:** Style consistency metrics, persona vector alignment
- **System Performance:** Memory usage, inference latency, throughput

**Final Deliverable:** Complete multimodal LLM system with audio projector, persona adaptation, and optimized deployment configuration

**Revised Total Estimated A100 GPU Hours:** Approximately 100-130 hours total (curriculum learning + LoRA training + DPO refinement)

#### Critical Path Dependencies (Simplified)

- All LoRA training can happen in parallel (Week 1)
- WQRM development needed before DPO (Week 2)  
- PersonaPlugs runtime development can happen in parallel with DPO (Week 2-3)
- No sequential dependencies between LoRA adapters

### 1.5. Resource Requirements

Our development setup centers around the Apple M1 Max for primary development and local testing, providing the computational power needed for model inference and initial validation. When it comes to serious training, we'll rely on NVIDIA A100 GPU access for focused training bursts, as estimated in our timeline.

The foundation of our system is the Google Gemma 3 4B model, specifically chosen because it hasn't undergone extensive RLHF processing. This gives us the raw stylistic capabilities we need for genuine personalization.

Our core techniques include LoRA for all adaptations, Direct Preference Optimization for quality refinement, quantization for deployment efficiency, and ExecuTorch for final deployment to ensure optimal performance on target hardware.

## 2. PersonaPlugs-Conditioned Architecture

### 2.1. Foundational Principle: LoRA-based Personalization and Adaptation

The entire "lil Pushkin" architecture is built upon the principle of using Low-Rank Adaptation (LoRA) for all model fine-tuning and specialization. The Google Gemma 3 4B base model's weights remain frozen. Personalization and new capabilities are introduced by training a series of LoRA layers.

This process begins with the creation of a **Foundational Persona LoRA**. This initial LoRA is trained on diverse, high-quality author corpora to imbue the base model with the ability to adapt to various writing styles. It serves as the common ancestor for all subsequent LoRAs, ensuring that stylistic understanding is a core component from the outset. At runtime, this LoRA (and its derivatives) will be conditioned by dynamically generated persona vectors from user documents (see Section 2.4: PersonaPlugs: Runtime Personalization Details) to achieve specific user personalization.

### 2.2. Persona-Conditioned Multimodal Architecture

**Audio Processing Stack:**

The audio processing architecture follows the proven Llama-AVSR methodology, starting with a frozen Wav2Vec2-XLS-R-300M encoder that converts audio waveforms into dense representations. The key innovation is our specialized MLP projector that bridges audio embeddings into Gemma's text space.

**PersonaPlugs Processing Stack:**

PersonaPlugs uses a similar MLP projector approach for user personalization. Document embeddings from user writing samples are generated using a high-quality embedding model (similar to RAG systems), then projected into Gemma's text space through a dedicated MLP projector trained on diverse textual data.

**Dual MLP Projector Architecture:**

Both projectors implement sophisticated two-layer architectures designed for stable training:
- Input stabilization through LayerNorm
- Expansion layer for rich representation learning
- Smooth activation with GELU
- Regularization through Dropout
- Projection to target dimensionality
- Output stabilization through LayerNorm

This architecture ensures stable gradients during training while providing sufficient capacity for complex mappings - both audio-to-text and document-embedding-to-text.

**Text Processing Integration:**

The text processing builds on Google Gemma 3 4B as our non-RLHF foundation, with Persona LoRA adapters trained on top for style conditioning. The audio projector outputs are concatenated with text embeddings before being fed into Gemma:

```
Audio Input → Wav2Vec2 → MLP Projector → [Audio Embeddings; Text Embeddings] → Gemma + Persona LoRA
```

**Unified Multimodal Processing:**

Our system handles both text and audio input through a unified approach that always includes persona conditioning. For text input, we combine PersonaVector with TextEmbedding and ContextEmbedding. Audio input follows the same pattern but uses ProjectedAudioEmbedding from our MLP projector.

The key advantage of this architecture is seamless integration - the MLP projector learns to map audio features into the same semantic space as text embeddings, allowing Gemma to process multimodal input naturally without architectural modifications.

**Curriculum Learning Integration:**

The training process follows a structured curriculum:
1. **Basic Transcription:** MLP projector learns audio→text mapping  
2. **Emotion Recognition:** Extended audio understanding with emotional labels
3. **Persona Integration:** LoRA training for personalized multimodal responses
4. **Task Specialization:** Specific capabilities (continuation, rephrasing) with audio input

### 2.3. Training Pipeline Strategy

The training strategy for "lil Pushkin" follows a **sequential MLP projector + LoRA approach**. We first train specialized MLP projectors to handle different input modalities, then train LoRA adapters for task-specific capabilities.

**Core Principle:** The Gemma 3 4B base model weights remain frozen throughout training. Different input modalities are handled through specialized MLP projectors, while task capabilities and personalization use LoRA adapters.

**Sequential Training Architecture:**

**Phase 1: MLP Projector Training**
- Audio MLP Projector: Curriculum learning from transcription to emotion recognition
- PersonaPlugs MLP Projector: Document embeddings to LLM space projection

**Phase 2: LoRA Adapter Training**  
- Persona LoRA: Style adaptation working with PersonaPlugs projector
- Task-Specific LoRAs: Specialized capabilities (continue, rephrase, etc.)

**Phase 3: System Integration & Refinement**
- DPO refinement across integrated components
- End-to-end optimization and deployment preparation

**Mixed-Precision Training Strategy** optimizes memory usage and training stability:

- **Gemma**: INT4 weights + BF16 activations (frozen, significant memory reduction)
- **Audio Encoder**: BF16 (frozen, memory efficient)  
- **MLP Projectors**: FP32 (trainable, ensures gradient stability)
- **LoRA Adapters**: FP32 (trainable, critical for quality)

This quantization strategy prevents gradient issues while maximizing memory efficiency for the frozen components.

```mermaid
graph TB
    subgraph "Base Model (Frozen)"
        G[Google Gemma 3 4B<br/>Non-RLHF Base<br/>INT4 weights + BF16]
        style G fill:#ffebee
    end
    
    subgraph "Audio Processing Pipeline"
        AE[Wav2Vec2-XLS-R-300M<br/>Audio Encoder<br/>BF16, Frozen]
        AMP[Audio MLP Projector<br/>2-Layer + LayerNorm<br/>FP32, Trainable]
        style AE fill:#e1f5fe
        style AMP fill:#f3e5f5
    end
    
    subgraph "PersonaPlugs Pipeline"
        EE[Document Embedding Model<br/>High-Quality Embeddings<br/>Frozen]
        PMP[PersonaPlugs MLP Projector<br/>2-Layer + LayerNorm<br/>FP32, Trainable]
        style EE fill:#e8f5e8
        style PMP fill:#f3e5f5
    end
    
    subgraph "Training Phases"
        P1[Phase 1: MLP Projector Training<br/>Audio: Curriculum Learning<br/>PersonaPlugs: General Projection]
        P2[Phase 2: LoRA Training<br/>Persona + Task-Specific]
        P3[Phase 3: DPO Refinement<br/>End-to-End Optimization]
        style P1 fill:#fff8e1
        style P2 fill:#f1f8e9
        style P3 fill:#fce4ec
    end
    
    subgraph "LoRA Adapters (FP32)"
        PL[Persona LoRA<br/>Style Adaptation<br/>Rank 64, Alpha 128]
        TL1[Continue LoRA<br/>Text Continuation]
        TL2[Rephrase LoRA<br/>Text Rephrasing]
        TL3[Other Task LoRAs<br/>As Needed]
        style PL fill:#fff3e0
        style TL1 fill:#e8f5e8
        style TL2 fill:#e8f5e8
        style TL3 fill:#e8f5e8
    end
    
    subgraph "Runtime Compositions"
        C1[Text Writing<br/>Base + PersonaPlugs + Persona + Task]
        C2[Audio Transcription<br/>Base + Audio Projector]
        C3[Personalized Audio<br/>Base + Both Projectors + All LoRAs]
        C4[Pure Autocomplete<br/>Base + Task LoRA Only]
        style C1 fill:#f0f8ff
        style C2 fill:#f0f8ff
        style C3 fill:#f0f8ff
        style C4 fill:#f0f8ff
    end
    
    subgraph "Quality Enhancement"
        W[Writing Quality<br/>Reward Model<br/>Multimodal Metrics]
        D[DPO Refinement<br/>Integrated Preferences]
        style W fill:#ffeaa7
        style D fill:#ffeaa7
    end
    
    AE --> AMP
    EE --> PMP
    
    P1 --> AMP
    P1 --> PMP
    P2 --> PL
    P2 --> TL1
    P2 --> TL2
    P2 --> TL3
    
    AMP --> G
    PMP --> G
    PL --> G
    TL1 --> G
    TL2 --> G
    TL3 --> G
    
    G --> C1
    G --> C2
    G --> C3
    G --> C4
    
    PMP --> C1
    PL --> C1
    TL1 --> C1
    
    AMP --> C2
    
    AMP --> C3
    PMP --> C3
    PL --> C3
    TL1 --> C3
    
    TL1 --> C4
    
    W --> D
    D --> PL
    D --> TL1
    D --> TL2
    D --> TL3
```
**Pipeline Stages:**

#### 2.3.1. Stage 1: Audio MLP Projector Training (Curriculum Learning)

**Basic Audio Transcription Training**

- **Goal:** Train the MLP projector to map audio embeddings into Gemma's text space for basic transcription tasks.
- **Base:** Google Gemma 3 4B (frozen) + Wav2Vec2-XLS-R-300M (frozen)
- **Trainable Component:** Audio MLP Projector only (FP32 for gradient stability)
- **Data:** High-quality transcription datasets from various sources and languages
- **Metrics:** Word Error Rate (WER), Real-Time Factor (RTF), gradient stability monitoring
- **Training Strategy:** Mixed-precision with GradScaler, projector in FP32, frozen components in BF16/INT4

**Emotion Recognition Integration**

- **Goal:** Extend projector capabilities to recognize and encode emotional content from speech.
- **Base:** Previous stage + trained Audio MLP projector
- **Data:** Emotional speech datasets with labeled emotional content
- **Additional Metrics:** Emotion Recognition Accuracy, emotional consistency scores
- **Training Approach:** Continued training of projector with emotional labels as additional supervision

#### 2.3.2. Stage 2: PersonaPlugs MLP Projector Training

**Document Embedding Projection Training**

- **Goal:** Train PersonaPlugs MLP projector to map document embeddings into Gemma's text space.
- **Base:** Gemma 3 4B (frozen) + pre-trained embedding model (frozen)
- **Trainable Component:** PersonaPlugs MLP Projector (FP32)
- **Data:** Diverse textual datasets for general text understanding and projection
- **Training Strategy:** Similar architecture to audio projector, optimized for text embedding inputs
- **Target Outcomes:** Effective projection of user document embeddings for personalization

#### 2.3.3. Stage 3: Persona LoRA Integration

**Persona Style Adaptation Training**

- **Goal:** Train persona LoRA to work with the PersonaPlugs projector for personalized responses.
- **Base:** Gemma 3 4B + trained PersonaPlugs MLP projector (frozen)
- **Trainable Component:** Persona LoRA (rank 64, alpha 128) in FP32
- **Data:** High-quality author corpora with diverse writing styles
- **Training Approach:** LoRA training on top of base model, coordinated with PersonaPlugs projector outputs
- **Target Outcomes:** Consistent personalization across different input types

#### 2.3.4. Stage 4: Task-Specific LoRA Specialization

**Task Adaptation Training**

- **Goal:** Train task-specific LoRAs for different writing capabilities.
- **Base:** Gemma 3 4B + trained projectors + Persona LoRA
- **Trainable Components:** Task-specific LoRAs (rank 64, alpha 128) in FP32
- **Data:** Task-specific datasets:
  - **Continue:** Text continuation examples
  - **Rephrase:** Rephrasing and style transfer examples
  - **Other tasks:** As needed for specific capabilities
- **Training Strategy:** Fine-tune task LoRAs while keeping projectors and persona LoRA frozen
- **Target Outcomes:** Specialized capabilities that work seamlessly with both projectors and personalization

#### 2.3.5. Stage 5: DPO Refinement

**Quality Enhancement with Multimodal DPO**

- **Goal:** Apply DPO to improve quality across all system components while maintaining capabilities.
- **Base:** Complete integrated system (Gemma + both projectors + all LoRAs)
- **Process:** Generate preference pairs from various inputs, apply quality scoring, DPO training
- **Data:** Preference pairs covering text, audio, and personalized content
- **Target Outcomes:** High-quality system with preserved multimodal and personalization capabilities

**Training Resource Estimates:**

- **Audio MLP Projector Training:** Curriculum learning phases require substantial compute
- **PersonaPlugs MLP Projector Training:** Moderate compute requirements for embedding projection  
- **Persona LoRA Integration:** Standard LoRA training resource needs
- **Task-Specific LoRAs:** Variable depending on number of tasks implemented
- **DPO Refinement:** Additional compute for preference optimization across all components

This training pipeline ensures stable progression from basic modality handling to complex integrated multimodal and personalized interactions.

### 2.4. PersonaPlugs: Runtime Personalization Details

PersonaPlugs is the methodology for dynamically conditioning the pre-trained LoRAs at inference time to adapt to an individual user's writing style. It uses a dedicated MLP projector to map document embeddings into Gemma's text space.

**Process for Generating and Applying Persona Vectors at Runtime:**

1. **User Document Indexing**: Text from user documents (.txt, .rtf, .docx, .pdf) created or opened in Killah is processed (with user consent and opt-out controls).

2. **Document Embedding Generation**: Documents are processed using a high-quality embedding model (similar to those used in RAG systems) to generate dense vector representations capturing semantic and stylistic content.

3. **Persona Vector Projection**: The document embeddings are passed through the trained PersonaPlugs MLP projector to map them into Gemma's text space, creating `PersonaVectors` that can be directly used by the model.

4. **Dynamic Persona Conditioning**: When the user invokes a Killah feature, relevant persona vectors from their indexed documents are retrieved and used to condition the model's responses, ensuring consistent personalization across different tasks and modalities.

5. **LLM Integration**: The projected persona vectors are provided as additional input to the LoRA-equipped Gemma model, typically concatenated with other inputs to guide style and content generation.

This runtime conditioning ensures that all interactions with "lil Pushkin" are personalized, leveraging the user's own data through the trained MLP projector to tailor the model's behavior without requiring retraining for each user.

## 3. Enhanced Multimodal Evaluation Metrics

Our evaluation framework extends traditional text quality assessment to handle audio-originated content and multimodal interactions, essential for validating the curriculum learning pipeline and audio-aware DPO training.

### 3.0. Audio-Aware Metric Categories

**Audio Processing Metrics** measure the quality of audio-to-text transformation and the stability of the MLP projector training:

- **Word Error Rate (WER)**: Transcription accuracy against ground truth
- **Real-Time Factor (RTF)**: Processing efficiency for real-time applications  
- **Audio-Text Semantic Consistency**: Semantic preservation from audio to text
- **Emotion Recognition Accuracy**: Consistency of emotional content detection

**Multimodal Quality Metrics** assess how well the system maintains quality when switching between or combining modalities:

- **Cross-Modal Style Consistency**: Whether personalization is maintained across text and voice input
- **Audio-Guided Writing Quality**: How well audio input guides high-quality text generation
- **Voice Command Recognition**: Accuracy of distinguishing dictation from commands

**Gradient Stability Metrics** monitor training health in the mixed-precision setup:

- **Projector Gradient Norms**: Ensuring MLP projectors train stably in FP32
- **LoRA Gradient Health**: Monitoring for NaN or exploding gradients
- **Memory Usage Tracking**: Validating quantization effectiveness

### 3.1. Understanding Our Three-Tier Metric System

Before diving into specific metrics, it's important to understand that we have three distinct categories of measurement, each serving a different purpose:

**Rigid Formula Metrics** are mathematical calculations we can compute directly from text without any learning. These include things like Type-Token Ratio, sentence length distributions, and n-gram overlap. They're fast, deterministic, and great for catching obvious issues, but they miss nuanced quality aspects.

**Learned Reward Model Metrics** come from our Writing Quality Reward Model (WQRM), which has learned to assess quality from human edit data. This model looks at edit patterns and learns what makes text better. It's particularly good at catching "AI slop" and understanding when text needs improvement, because it's seen thousands of examples of humans fixing problematic text.

**DDPO and Evaluation Metrics** are what we use during training (DDPO) and final assessment. These combine both rigid formulas and learned components to create comprehensive scores. During DDPO training, we generate multiple responses, score them with our full metric suite, and use the scores to create preference pairs (best vs. worst). For evaluation, we use these same metrics to benchmark our model's performance against baseline models and human writing.

### 3.2. Automated Evaluation Metrics

#### 3.2.1. Style Consistency and Personalization Metrics (Rigid Formulas)

These are mathematical measures we can calculate directly from the text. They're fast and reliable, but they only capture surface-level patterns.

**Persona Vector Alignment Score** measures how well generated text aligns with the user's persona vector by computing cosine similarity between the embedding of generated text and the target persona vector. Higher scores indicate better style matching. This is computed as a simple dot product between normalized vectors.

**N-gram Overlap with User Corpus** calculates the overlap of distinctive n-grams (2-grams through 5-grams) between generated text and the user's historical writing. This metric tracks how well the model adopts user-specific vocabulary patterns and phrasal choices. It's essentially counting shared word sequences and computing percentages.

**Stylistic Feature Matching** quantifies alignment across multiple dimensions. Sentence Length Distribution uses KL divergence between sentence length distributions of generated versus user text. Vocabulary Richness compares Type-Token Ratio (TTR) and MTLD (Measure of Textual Lexical Diversity). Syntactic Complexity looks at average parse tree depth using dependency parsing. Formality Score measures the ratio of formal versus informal language markers. All of these are computed using straightforward statistical formulas.

#### 3.2.2. Text Quality and Coherence Metrics (Mixed: Rigid + Learned)

This is where things get more interesting. Some of these metrics are computed with simple formulas, but others require learned models to assess properly.

**Semantic Coherence** uses both approaches. Cosine Similarity Between Adjacent Sentences is a rigid formula that measures semantic flow by computing embedding similarity between consecutive sentences. Too high indicates repetition, too low suggests incoherence. But Mutual Information Between Text Segments is more sophisticated—it evaluates how much information each sentence provides given the previous context, requiring learned models to assess properly.

**Lexical Diversity and Richness** mostly uses rigid formulas. MTLD (Measure of Textual Lexical Diversity) is a robust statistical measure of vocabulary diversity that accounts for text length variations. Entropy-based Measures compute word-level entropy to assess vocabulary distribution richness. Novel N-gram Ratio calculates the percentage of n-grams in generated text that are novel relative to the training data, indicating creativity versus memorization.

**Structural Quality** combines both approaches. Parse Tree Validity uses standard parsing tools to measure the percentage of generated sentences that parse correctly without syntax errors—this is a rigid formula. But Entity Consistency is more complex. It tracks named entities (characters, places, objects) and verifies their attributes remain consistent throughout generated text using relationship graphs. This requires some learned understanding of what constitutes consistent character behavior.

#### 3.2.3. Anti-"AI Slop" and Literary Quality Metrics (Reward Model Learned)

This is where our Writing Quality Reward Model (WQRM) really shines. These metrics can't be computed with simple formulas because they require understanding what constitutes "good" versus "bad" writing based on human editing patterns.

**Edit-Distance Quality Assessment** is the core of our learned approach. It quantifies how much editing would be required to transform generated text into high-quality writing, using patterns learned from expert edit traces. Our WQRM has seen thousands of examples where humans took mediocre text and made it better, so it can predict which parts of new text would likely need similar improvements.

**Generic Phrase Detection** is partially learned, partially rigid. We maintain databases of common "AI tells" like overuse of transition phrases ("Moreover," "Furthermore," "In conclusion"), repetitive sentence structures, corporate-speak, and hedging language ("it's worth noting," "it's important to"). The detection itself uses pattern matching (rigid), but the penalty weights are learned from examples of human editors removing these phrases.

**Cliché and Trope Detection** works similarly—we use pattern matching against databases of common phrases, plot tropes, and overused expressions, but the severity scoring is learned from editorial decisions.

**Literary Quality Dimensions** are almost entirely learned metrics. Originality and Surprise can be partially measured through novel n-gram ratios (rigid), but truly assessing whether something is creative versus just random requires learned understanding. Emotional Engagement looks at sentiment dynamics and emotional arc development, which requires understanding narrative structure. Stylistic Sophistication involves balancing complexity and clarity while avoiding both oversimplification and purple prose—this definitely requires learned judgment. Show versus Tell Ratio quantifies descriptive versus expository writing balance, again requiring learned understanding of what constitutes "showing" versus "telling."

**Compression-based Aesthetic Metrics** are rigid formulas that measure information density using text compression ratios. Well-crafted text should pack more meaning into fewer words, achieving high information content without verbosity. This is computed using LZMA compression.

* **Text Aesthetics and Interestingness Features:** Based on Kindle "popular highlights" analysis with precise mathematical formulations:

  * **Word Repetition Emphasis (W1):** Average positional difference weighted count of word repetitions—closer repetitions indicate stronger emphasis:

  $$
  W1(P) = \frac{2}{N(N-1)} \sum_{i=1}^{N} \sum_{j=i+1}^{N} \frac{1(w_i = w_j)}{j-i}
  $$

  Where $1(w_i = w_j)$ is indicator function for word matches, weighted by inverse positional distance.

  * **Average Word Length (W2):** Measures sophistication through word length patterns:

  $$
  W2(P) = \frac{1}{N} \sum_{i=1}^{N} \text{len}(w_i)
  $$

  * **Topic Diversity (T1):** Captures broad thematic appeal by measuring topic class mismatches:

  $$
  T1(P) = \frac{2}{N(N-1)} \sum_{i=1}^{N} \sum_{j=i+1}^{N} \frac{1[z(w_i) \neq z(w_j)]}{j-i}
  $$

  Where $z(w)$ is LDA topic class of word $w$, weighted by proximity of diverse concepts.

  * **Topic Abstractness (T2):** Measures philosophical/abstract content through topic representativeness:

  $$
  T2(P) = \frac{1}{N} \sum_{i=1}^{N} \max_k \phi_k(w_i)
  $$

  Lower values indicate more abstract, less topic-representative words (philosophical content).

  * **Part-of-Speech Richness (POS):** Emphasis through adjectives and adverbs:

  $$
  POS(P) = \frac{1}{N} \sum_{i=1}^{N} (\text{adjectives}_i + \text{adverbs}_i)
  $$

  * **Sentiment Contrast (SENT):** Antithesis detection through sentiment polarity differences:

  $$
  SENT(P) = \frac{2}{N(N-1)} \sum_{i=1}^{N} \sum_{j=i+1}^{N} \frac{|s(w_i) - s(w_j)|}{j-i}
  $$

  Where $s(w)$ is SentiWordNet sentiment value, emphasizing nearby contrasts.

  * **Semantic Distance Variance (SD1, SD2):** Measures conceptual diversity through DISCO similarity:

  $$
  SD_k(P) = \frac{2}{N(N-1)} \sum_{i=1}^{N} \sum_{j=i+1}^{N} \frac{|ds_k(w_i) - ds_k(w_j)|}{j-i},\quad k \in \{1,2\}
  $$

  Where $ds_1$ is first-order (collocation) and $ds_2$ is second-order (distributional) similarity.

  9. **Continue Task:**
  * **BLEU/ROUGE Scores:** Against human-written continuations of the same prompts
  * **Narrative Coherence Score:** Measures plot consistency and logical flow
  * **Style Preservation:** Quantifies how well the continuation maintains the original text's style

* **Rephrase Task:**
  * **BLEURT Scores:** Semantic similarity to target rephrasing
  * **Meaning Preservation:** Embedding-based similarity to ensure semantic content is retained
  * **Style Transfer Effectiveness:** Measures successful adoption of target style while preserving meaning

* **Story Generation Task:**
  * **Plot Originality:** Measures uniqueness against existing story databases
  * **Character Development Consistency:** Tracks character traits and development arcs
  * **Narrative Structure Adherence:** Evaluates proper story structure (exposition, rising action, climax, resolution)

#### 3.1.7. Creativity Detection Metrics

Based on research into automatic creativity detection that distinguishes creative from non-creative text using novelty and quality criteria:

* **Lexical Creativity Measures:**
  * **Type-to-Token Ratio (TTR):** Vocabulary richness measure indicating creative word choice diversity:

  $$
  TTR = \frac{C_{unique}}{n}
  $$

  Where $C_{unique}$ is unique words and $n$ is total words. Higher TTR suggests more creative vocabulary usage.

  * **Word Norms Fraction:** Measures text "usualness" by detecting conventional word associations:

  $$
  WNF = \frac{C_{norm}(x,y)}{n}
  $$

  Where $C_{norm}(x,y)$ counts word pairs appearing in Free Association Norms database (72,176 pairs).

* **Semantic Creativity Assessment:**
  * **Google Similarity Distance:** Measures semantic novelty using web-based word co-occurrence:

$$
GSD(x,y) = \frac{\max[\log f(x), \log f(y)] - \log f(x,y)}{\log M - \min[\log f(x), \log f(y)]}
$$

Where $f(x)$ is page counts containing term $x$, $f(x,y)$ is pages containing both terms, and $M$ is total indexed pages (50 billion).

  * **Explicit Semantic Analysis (ESA):** Wikipedia-based semantic relatedness computation:
    * Uses Wikipedia articles as concept vectors
    * Represents text meaning in terms of Wikipedia concepts
    * Applies TFIDF weighting and inverted indexing
    * Computes semantic relatedness via cosine similarity
    * Detects unconventional concept combinations indicating creativity

  * **WordNet Similarity:** Lexical database-based similarity measurement:
    * Uses WordNet lexical ontology structure
    * Calculates shortest path between word senses
    * Identifies semantic relationships and distances
    * Lower similarity scores may indicate creative word usage

* **Structural Creativity Indicators:**
  * **Number of Named Entities:** Raw count of named entities in text:

$$
m5 = \text{total named entities in text}
$$

  * **Named Entity Diversity Score:** Proportion of distinct named entities indicating creative reference patterns:

$$
NE_{Score} = \frac{\text{distinct named entities}}{\text{total named entities}}
$$

  * **Coherence Measure:** Text coherence assessment through sentence similarity:

$$
\text{Coherence} = \frac{|\{(d_i, d_j) : \cos(d_i, d_j) \geq \tau\}|}{|D|^2}
$$

  Where $\tau$ is similarity threshold (0.05), $D$ is document set. Creative texts show optimal coherence balance—not too repetitive, not too scattered.

  * **Latent Semantic Analysis (LSA) Creativity Measures:**
  * **Average Similarity Between Adjacent Sentences:**

$$
m_{9a} = \frac{1}{n-1} \sum_{i=1}^{n-1} s_{i,i+1}
$$

  Where $s_{i,i+1}$ is similarity from SVD-reduced sentence matrix.

  * **Average Similarity Between All Sentences:**

$$
m_{9b} = \frac{2}{n(n-1)} \sum_{i=1}^{n} \sum_{j=i+1}^{n} s_{i,j}
$$

  * **Average Cosine Similarity Between Adjacent Sentences:**

$$
m_{9c} = \frac{1}{n-1} \sum_{i=1}^{n-1} \cos(v_i, v_{i+1})
$$

  Where $v_i$ are sentence vectors from SVD decomposition.

  * **Average Cosine Similarity Between All Sentences:**

$$
m_{9d} = \frac{2}{n(n-1)} \sum_{i=1}^{n} \sum_{j=i+1}^{n} \cos(v_i, v_j)
$$

* **Creativity Classification Model:**
  * **Stepwise Logistic Regression:** Combines all nine measures using stepwise feature selection
  * **Training Data:** Creative texts (satirical news from The Onion) vs non-creative texts (conventional news articles)
  * **Performance:** Achieves ~80% accuracy in distinguishing creative from conventional text
  * **Limitation:** May detect satire rather than pure creativity; requires validation across different domains and text genres

### 3.2. Establishing Evaluation Baselines

To make the automated metrics meaningful, it's crucial to establish baselines. Without baselines, it's hard to determine if a score is "good" or "bad." Here’s a methodological approach:

1.  **Select Reference Corpora:**
  * **General Quality Corpus:** Choose a small, diverse corpus of high-quality text in your target languages (EN/RU) that is NOT part of your training data. This could include well-regarded news articles, book excerpts, or essays (e.g., 5-10 documents, a few thousand words each). This corpus helps establish what "good general writing" looks like according to your metrics.
  * **Target Style Corpus (User/Author Data):** If you are trying to emulate a specific author or a user's style, use a sample of their writing (again, not from the training set) as a reference. This is vital for personalization metrics. If no specific user data is available initially, use texts from authors whose style you aim to enable the model to adopt.
  * **Base Model Output (Optional but Recommended):** Generate outputs from the base Gemma 3 4B model (without any LoRAs) using the same prompts you'll use for your model. This provides a "vanilla" baseline.

2.  **Calculate Metrics on Reference Corpora:**
  * Run your chosen automated metrics (e.g., MTLD, sentence length distribution, n-gram overlap if comparing to a specific style) on these reference corpora.
  * For example, calculate the average MTLD and sentence length statistics for your "General Quality Corpus." For the "Target Style Corpus," calculate n-gram profiles and stylistic features.

3.  **Interpret Your Model's Scores:**
  * When you evaluate your trained "lil Pushkin" model, compare its scores against these baselines.
  * **Style Consistency:** Does `Persona Vector Alignment Score` improve significantly compared to the base model when conditioned on a persona? Does the `N-gram Overlap` with a target style corpus increase? Do stylistic features (sentence length, formality) move closer to the target style?
  * **Text Quality:** Are metrics like MTLD, coherence scores, and anti-"AI Slop" indicators comparable to or better than the "General Quality Corpus" and the base model?
  * **Task-Specific Metrics:** For tasks like "Continue," how do BLEU/ROUGE scores compare when your model's output is evaluated against human continuations, versus perhaps a simpler baseline continuation strategy?

4.  **Iterate:**
  * Baselines help you understand if your model is improving in the desired directions during development. If a metric is unexpectedly low compared to a relevant baseline, it might indicate an area for further tuning or data refinement.

This process provides a quantitative way to ground your evaluation and track progress beyond subjective impressions. Start with a few key metrics and expand as needed.

### 3.3. Human Evaluation Protocols (Summary - keep concise)

* **Paired Comparisons:** Raters choose better output between Model A vs. Model B.
* **Likert Scales:** Rate outputs on fluency, coherence, style match.
* **Task Success Rate:** Can users achieve goals with the model?

### 3.4. Real-World Usage Analytics (Summary - keep concise)

* Track feature adoption, user retention, qualitative feedback.

## 4. ОБНОВЛЕННАЯ МОДУЛЬНАЯ АРХИТЕКТУРА LORA

### 4.1. Ключевое изменение в подходе

**Вместо последовательного обучения LoRA адаптеров**, где каждый следующий LoRA тренируется поверх предыдущего, мы используем **независимое обучение** с последующей **динамической композицией**.

### 4.2. Новая схема тренировки

#### Этап 1: Независимое обучение LoRA адаптеров (параллельно)

1. **Persona LoRA** - тренируется на базовой Gemma 3 4B для изучения стилей
2. **Task LoRA (Continue)** - тренируется на базовой Gemma 3 4B для продолжения текста
3. **Task LoRA (Rephrase)** - тренируется на базовой Gemma 3 4B для перефразирования
4. **Audio LoRA** - тренируется на базовой Gemma 3 4B для обработки аудио

#### Этап 2: Композиция во время инференса

```python
# Пример композиции для разных задач
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b")

# Для персонализированного продолжения текста:
model = PeftModel.from_pretrained(base_model, "persona_lora/")
model = PeftModel.from_pretrained(model, "task_continue_lora/")

# Для голосовой диктовки с персонализацией:
model = PeftModel.from_pretrained(base_model, "persona_lora/")
model = PeftModel.from_pretrained(model, "audio_lora/")

# Полная мультимодальная система:
model = PeftModel.from_pretrained(base_model, "persona_lora/")
model = PeftModel.from_pretrained(model, "task_continue_lora/")
model = PeftModel.from_pretrained(model, "audio_lora/")
```

### 4.3. Преимущества новой архитектуры

1. **Отсутствие катастрофического забывания** - каждый LoRA сохраняет свои способности
2. **Параллельное обучение** - все LoRA могут тренироваться одновременно
3. **Гибкость композиции** - можно динамически включать/выключать способности
4. **Модульность** - легко добавлять новые способности без переобучения
5. **Отладка** - проблемы изолированы в конкретных LoRA

### 4.4. Техническая реализация

**Во время обучения:**
- Каждый LoRA тренируется независимо на базовой модели
- Никаких зависимостей между LoRA адаптерами
- Можно использовать разные GPU для разных LoRA

**Во время инференса:**
- LoRA адаптеры загружаются последовательно
- Математически их веса просто складываются
- Можно менять композицию на лету

### 4.5. Обновленная диаграмма архитектуры

```mermaid
graph TB
    Base[Google Gemma 3 4B Base] --> P[Persona LoRA]
    Base --> T1[Task Continue LoRA]
    Base --> T2[Task Rephrase LoRA]
    Base --> A[Audio LoRA]
    
    P -.-> C1[Writing Mode<br/>Persona + Task]
    T1 -.-> C1
    
    P -.-> C2[Voice Mode<br/>Persona + Audio]
    A -.-> C2
    
    P -.-> C3[Full Mode<br/>All LoRAs]
    T1 -.-> C3
    A -.-> C3
```

Это кардинально меняет нашу стратегию разработки и делает систему гораздо более гибкой!

## 4. Modular LoRA Architecture: Technical Foundation and Advantages

### 4.1. Why Modular LoRA Architecture is Critical

**The Problem with Sequential LoRA Training:**
Traditional approaches train LoRA adapters sequentially, where each new adapter builds upon the previous one. This creates a fundamental issue known as **catastrophic forgetting** - when a neural network learns new tasks, it tends to "forget" previously learned capabilities as the weights shift to accommodate new patterns.

**Example of Sequential Training Problems:**
```
Base Model → Persona LoRA → Audio LoRA → Task LoRA
```
In this approach:
- Training Audio LoRA on top of Persona LoRA can degrade the model's understanding of writing style
- Adding Task LoRA can interfere with both persona and audio capabilities
- Each stage risks losing previously acquired skills
- Testing and debugging becomes extremely difficult as capabilities are intertwined

**Our Solution: Independent Training with Runtime Composition**
Instead, "lil Pushkin" uses **modular, independent LoRA training** where each adapter is trained separately on the base model and then composed at runtime:

### 4.2. Independent LoRA Training Strategy

#### Stage 1: Parallel Independent Training
Each LoRA adapter is trained independently on the base Gemma 3 4B model:

1. **Persona LoRA:** `Base Model + Style Dataset` → Pure style adaptation
2. **Audio LoRA:** `Base Model + Audio-Text Dataset` → Pure multimodal processing  
3. **Task LoRA (Continue):** `Base Model + Continuation Dataset` → Pure task capability
4. **Task LoRA (Rephrase):** `Base Model + Rephrase Dataset` → Pure task capability

#### Stage 2: Runtime Composition
At inference time, we dynamically load and compose the required LoRA adapters:

```python
# Runtime composition examples
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b")

# For personalized text writing (default interactive mode):
model = PeftModel.from_pretrained(base_model, "persona_lora/")
model = PeftModel.from_pretrained(model, "audio_lora/")  # Always present for voice feedback
model = PeftModel.from_pretrained(model, "task_continue_lora/")

# For voice dictation (full multimodal):
model = PeftModel.from_pretrained(base_model, "persona_lora/")
model = PeftModel.from_pretrained(model, "audio_lora/")
model = PeftModel.from_pretrained(model, "task_dictation_lora/")

# For pure autocomplete only (minimal composition):
model = PeftModel.from_pretrained(base_model, "task_autocomplete_lora/")
```

### 4.3. Technical Implementation: How LoRA Composition Works

**Mathematical Foundation:**
LoRA adapters modify the original model weights through low-rank decomposition:
```
W_modified = W_original + B × A
```
Where B and A are the low-rank matrices learned during LoRA training.

**Runtime Composition:**
When multiple LoRA adapters are loaded, their modifications are additive:
```
W_final = W_original + B₁×A₁ + B₂×A₂ + B₃×A₃
```
This allows us to combine capabilities without interference, as each LoRA operates in its own parameter subspace.

### 4.4. Key Advantages of Modular LoRA Architecture

#### 1. **Elimination of Catastrophic Forgetting**
- Each LoRA maintains its specialized knowledge independently
- No risk of new training degrading existing capabilities  
- Stable, predictable behavior across all use cases

#### 2. **Parallel Development and Training**
- All LoRA adapters can be trained simultaneously on different GPUs
- Dramatically reduces development time
- Independent testing and validation of each capability

#### 3. **Dynamic Flexibility**
- Runtime composition allows for context-aware capability selection
- Can optimize performance by loading only required adapters
- Easy to experiment with different capability combinations

#### 4. **Modular Debugging and Quality Control**
- Issues can be isolated to specific LoRA adapters
- Independent quality assessment for each capability
- Easier troubleshooting and improvement cycles

#### 5. **Scalable Architecture**
- New capabilities can be added without retraining existing adapters
- Version control for individual capabilities
- Easy deployment of capability updates

### 4.5. Why Other Approaches Fail: Catastrophic Forgetting Explained

**What is Catastrophic Forgetting?**
Catastrophic forgetting occurs when neural networks lose previously learned information upon learning new tasks. This happens because:

1. **Weight Interference:** New learning modifies shared weights, disrupting previous patterns
2. **Gradient Conflicts:** Gradients from new tasks can directly oppose previous learning
3. **Representation Drift:** Internal representations shift to accommodate new data, losing old associations

**Real-World Impact in Sequential LoRA Training:**
- A model trained on writing style (Persona LoRA) then trained on audio processing may lose its stylistic consistency
- Task-specific training can degrade the model's ability to maintain persona characteristics
- Audio processing capabilities might interfere with text generation quality

**Why Our Modular Approach Prevents This:**
- Each LoRA is trained on a frozen base model, preserving the original capabilities
- No weight interference between different capabilities during training
- Runtime composition maintains all learned behaviors simultaneously
- Each capability operates in its dedicated parameter space without conflicts

### 4.6. Persona and Audio LoRA as Core Components

**Design Principle:** In "lil Pushkin", **Persona LoRA and Audio LoRA are architectural foundations**, not optional features:

- **Persona LoRA** ensures all outputs match the user's writing style and preferences
- **Audio LoRA** provides consistent voice interaction capabilities and audio understanding
- These adapters are present in **all interactive scenarios** except pure autocomplete
- This creates a consistent, personalized, and multimodal user experience

**Implementation Strategy:**
```python
# Default LoRA composition for interactive features
DEFAULT_LORAS = ["persona_lora", "audio_lora"]
TASK_LORAS = {
    "continue": "task_continue_lora",
    "rephrase": "task_rephrase_lora", 
    "dictation": "task_dictation_lora",
    "autocomplete": "task_autocomplete_lora"  # Only task LoRA, no persona/audio
}

def compose_model_for_task(task_type):
    loras_to_load = DEFAULT_LORAS.copy()
    if task_type != "autocomplete":
        loras_to_load.append(TASK_LORAS[task_type])
    else:
        loras_to_load = [TASK_LORAS[task_type]]  # Minimal composition for speed
    
    return load_composed_model(loras_to_load)
```

This architecture ensures that "lil Pushkin" delivers consistent personalization and voice integration across all meaningful user interactions while maintaining optimal performance for high-frequency operations like autocomplete.

## 5. Min-p Sampling: Novel Decoding Strategy for Creative Writing

### 5.1. Implementation of Min-p Sampling for PersonaPlugs

Based on research into min-p sampling, "lil Pushkin" incorporates this decoding strategy, optimized for creative and coherent outputs, aiming to avoid pitfalls of traditional temperature and top-k sampling.

**Min-p Sampling Methodology:**

* **Core Principle:** Min-p sampling dynamically sets a probability threshold for token selection. It filters the vocabulary to include only tokens whose probabilities $P(\text{token})$ satisfy $P(\text{token}) \ge p \times P_{\text{max}}$, where $P_{\text{max}}$ is the probability of the most likely token in the distribution, and $p$ is the min-p value (a hyperparameter, typically around 0.1). Tokens below this threshold are excluded from sampling.

* **Process:**
  1.  Obtain the logits (raw output scores) from the model for the next token.
  2.  Convert logits to probabilities (e.g., using softmax).
  3.  Identify $P_{\text{max}}$, the highest probability among all tokens.
  4.  Calculate the threshold as $p \times P_{\text{max}}$.
  5.  Create a new distribution containing only tokens whose original probability meets or exceeds this threshold.
  6.  Renormalize these probabilities.
  7.  Sample from this filtered and renormalized distribution (optionally applying temperature scaling before sampling).

* **Advantages over Temperature/Top-k:**
  * **Coherence Preservation:** Aims to maintain logical flow by filtering out genuinely unlikely tokens rather than just truncating the distribution (top-k) or excessively flattening it (high temperature).
  * **Context Sensitivity:** The effective size of the candidate pool for sampling automatically adjusts based on the model's confidence. In contexts where the model is very certain about the next token (high $P_{\text{max}}$), the threshold is higher, leading to more focused sampling. In ambiguous contexts (lower $P_{\text{max}}$), the threshold is lower, allowing for more diverse and potentially creative choices.

**Integration with PersonaPlugs:**

* **Persona-Conditioned Thresholds:** The min-p value $p$ can be adjusted based on user writing style. For instance, users with a more creative or experimental style might benefit from slightly lower min-p values (e.g., 0.05-0.08) to encourage more diverse outputs. Users preferring more formal or predictable text might use higher values (e.g., 0.12-0.15). These preferences could potentially be learned from analyzing the user's document corpus.
* **Task-Specific Tuning:**
  * **Continue Task:** A balanced min-p (e.g., 0.08-0.12) might be suitable to ensure coherence with the preceding text while allowing for natural and creative continuation.
  * **StoryGen Task:** Lower min-p values (e.g., 0.05-0.10) could be explored to foster more imaginative narrative choices, while still relying on the underlying probability distribution to avoid complete incoherence.
* **Dynamic Adjustment:** While more advanced, min-p values could potentially be adjusted dynamically during generation based on context uncertainty or specific user preferences learned over time.

## 6. Audio-Visual Speech Recognition Integration

### 6.1. Audio-Based Speech Recognition Architecture

Drawing from state-of-the-art research in instruction-following speech recognition (Lai et al.) and audio speech recognition capabilities, "lil Pushkin" implements an audio-focused approach that integrates speech understanding with persona conditioning.

**Core Architecture Principles:**

* **Listen-Attend-Spell Foundation:** Uses proven LAS model architecture adapted for instruction-following capabilities, enabling natural language commands mixed with dictation
* **Audio Token Integration:** Audio tokens are processed alongside text tokens in the LLM, with frozen encoders and trainable LoRA projectors
* **Persona-Aware Speech Processing:** All audio understanding is conditioned on user communication patterns and voice characteristics

**Technical Implementation:**

```text
Audio Input → Conformer Encoder → Compression (K=3-4) → Linear Projector → Audio Tokens
Combined → [PersonaVector; AudioTokens; TextTokens] → Persona-Audio LoRA → Gemma 3 4B
```

**Training Data Requirements (Based on Llama-AVSR Results):**

* **Minimum Effective Dataset:** 30 hours labeled audio-visual data achieves WER ~28% for visual-only
* **Recommended Dataset:** 433 hours achieves WER ~1.3% for audio-visual recognition
* **Optimal Dataset:** 1,756 hours achieves state-of-the-art WER 0.77% for audio-visual tasks

**Key Training Insights:**

* **LoRA Parameter Efficiency:** Only 42-57M trainable parameters achieve SOTA results (vs. 325-570M for full fine-tuning)
* **Compression Rate Optimization:** K=3-4 for audio, K=2 for video provides optimal performance-efficiency trade-off
* **Encoder Selection:** Whisper-medium for audio + AV-HuBERT Large for video provides best multimodal performance

### 6.2. Instruction-Following Speech Capabilities

**Free-Form Instruction Processing:**

* **Natural Language Commands:** "Transcribe the first half and then turn off listening" or "Listen carefully, replace all instances of 'the' with 'Qokka'"
* **Privacy-First Design:** Selective transcription based on user instructions provides additional privacy layer
* **Context-Aware Processing:** Model distinguishes between dictation intent and command intent based on persona patterns

**Implementation Strategy:**

Our implementation follows a three-stage approach. Base training begins with LibriSpeech combined with instruction-following datasets to establish fundamental speech-text-instruction understanding. This foundation allows the model to process both spoken language and commands effectively.

Persona integration comes next, where we further train the model with user-specific voice patterns and command preferences. This stage teaches the model to recognize not just what users say, but how they typically communicate and what commands they prefer.

Finally, task specialization fine-tunes the system for writing-assistant specific commands like continue, rephrase, and story generation. This ensures that speech recognition is optimized for our specific use cases rather than general transcription.

### 6.3. Online Speech Recognition Optimization

Based on Apple's research on online LAS models (Hsiao et al.), incorporating techniques for low-latency, real-time speech processing:

**Silence Modeling Approach:**

Our approach to handling silence in speech recognition addresses a key challenge in online scenarios. We use explicit silence tokens, inserting them into training data to prevent the model from prematurely detecting end-of-sentence during natural pauses in speech.

The system handles asynchronous decoding, which is crucial since LAS models face the fundamental challenge that input consumption and output generation aren't synchronized in real-time scenarios. Our optimizations achieve 12% lower latency compared to conventional neural network HMM hybrids.

This latency reduction comes from better handling of silence regions and more efficient attention mechanisms that don't wait unnecessarily for additional input when the speaker is simply pausing.

**Technical Implementation:**

```python
# Silence token insertion during training data preparation
def insert_silence_tokens(transcript, alignment, silence_duration_frames=100):
  """Insert <SIL> tokens for silence segments longer than threshold"""
  tokens = []
  for segment in alignment:
    if segment.label == 'silence' and segment.duration > silence_duration_frames:
    num_silence_tokens = segment.duration // silence_duration_frames
    tokens.extend(['<SIL>'] * num_silence_tokens)
    else:
    tokens.append(segment.label)
  return tokens
```

**Production Deployment Considerations:**

For production deployment, we implement streaming buffer management using a sliding window approach that enables continuous audio processing without gaps or memory buildup. The attention mechanism uses MoChA (Monotonic Chunkwise Attention) specifically designed for online operation, ensuring low latency while maintaining accuracy.

Error recovery focuses on robust handling of silence regions and audio edge cases. This includes dealing with background noise, microphone issues, and the natural variations in how people speak, ensuring the system remains reliable in real-world usage scenarios.

## 7. Controllable Generation and PersonaPlugs Integration

### 7.1. Compute-Optimal Training Scaling

Based on Chinchilla scaling laws (Hoffmann et al.), optimize training data and compute allocation for different stages:

**Scaling Law Application:**

* **Base Model Scale:** Gemma 3 4B parameters requires ~80B tokens for compute-optimal training (already provided)
* **LoRA Training Scale:** With 42-57M trainable parameters, compute-optimal dataset size is 840M-1.14B tokens
* **Fine-tuning Ratios:** Each specialized LoRA requires ~10-20% of base training data (8-16B tokens) for optimal performance

**Stage-Specific Data Requirements:**

* **Foundational Persona LoRA:** 10-15B tokens of diverse author corpora (equivalent to ~5,000-7,500 books)
* **Persona-Audio LoRA:** 1,000-2,000 hours of transcribed audio-text pairs (~1.5-3B tokens)
* **Task-Specific LoRAs:** 2-5B tokens per task from high-quality task-specific datasets
* **DPO Refinement:** 100K-1M preference pairs per specialized LoRA

**Compute Budget Allocation:**

```text
Total Compute Budget (A100 hours): 175-350 hours
├── Foundational Persona LoRA: 30-60 hours (17-34%)
├── Persona-Audio LoRA: 40-80 hours (23-46%)  
├── Task-Specific LoRAs: 60-120 hours (34-69%)
└── DPO Refinement: 45-90 hours (26-51%)
```

**Efficiency Optimizations:**

* **Gradient Checkpointing:** Reduce memory usage by 40-50% with minimal computation overhead
* **Mixed Precision Training:** FP16 training reduces memory and increases throughput by 1.5-2x
* **Data Loading Optimization:** Efficient TFRecord format and parallel data loading prevent I/O bottlenecks

## 8. Advanced Training Techniques and Hyperparameters

### 8.1. LoRA Optimization Based on Research Insights

**Intruder Dimension Mitigation (Shuttleworth et al.):**

* **Higher Rank Usage:** r=64 instead of typical r=8-16 to reduce orthogonal singular vectors
* **Rank Stabilization:** Monitor cosine similarity between fine-tuned and pre-trained singular vectors
* **A-Matrix Freezing:** Optionally freeze LoRA A matrix during training to minimize intruder dimensions
* **Scaling Factor:** α = 2r (α = 128 for r=64) to maintain stable training dynamics

**Advanced LoRA Training Configuration:**

```python
lora_config = {
  "r": 64,
  "lora_alpha": 128,
  "target_modules": ["k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "rank_stabilization": True,
  "intruder_threshold": 0.6,
  "freeze_a_matrix": False  # Set to True if intruder dimensions exceed threshold
}
```

**Training Hyperparameters (Optimized for Gemma 3 4B):**

* **Learning Rate:** 2e-4 with cosine annealing schedule
* **Warmup Steps:** 500 steps (2% of total training)
* **Batch Size:** 32-64 depending on available memory
* **Gradient Clipping:** 1.0 to prevent instability
* **Weight Decay:** 0.01 for regularization
* **Optimizer:** AdamW with β₁=0.9, β₂=0.95, ε=1e-8

### 8.2. Audio Training Specifications

**Encoder Configuration:**

* **Audio Encoder:** Whisper-medium (24 layers, 1024 hidden, 16 attention heads)
* **Compression Rates:** K=3 for audio features
* **Projector Architecture:** Two linear layers with ReLU activation (~15M parameters)

**Audio Training Parameters:**

* **Audio Sample Rate:** 16kHz with 25ms window, 10ms shift
* **Sequence Length:** 6 seconds (150 audio frames, 150 video frames after compression)
* **Data Augmentation:** Horizontal flipping, random cropping, adaptive time masking

**Dataset Specifications:**

* **LibriSpeech:** 960 hours English read speech
* **Common Voice:** 1,400+ hours multilingual crowdsourced speech  
* **VoxCeleb2:** 1,000+ hours celebrity speech (with Whisper transcriptions)
* **LRS3:** 433 hours lip-reading dataset for audio-visual training
* **Russian Speech Corpora:** 500+ hours for multilingual support

### 8.3. Quality Enhancement Pipeline Details

**Multi-Stage Quality Assessment:**

1. **Initial Generation:** Generate 3-5 candidate responses using different sampling strategies
2. **Chain-of-Thought Editing:** Apply learned editing patterns to improve each candidate
3. **Comprehensive Scoring:** Evaluate using the 8-component WQRM scoring system
4. **Selection and Refinement:** Choose highest-scoring output with optional iterative improvement

**Edit Pattern Training Data:**

* **Literature Edit Traces:** Before/after pairs from professional literary editing
* **Academic Writing Improvements:** Scholarly article revision patterns
* **Creative Writing Refinements:** Workshop-style writing improvement examples
* **Technical Documentation Edits:** Clarity and conciseness improvement patterns

**Real-Time Quality Optimization:**

```python
def enhanced_generation_pipeline(prompt, persona_vector, task_type):
  candidates = []
  
  # Generate multiple candidates with different strategies
  for strategy in ['min_p_0.05', 'min_p_0.1', 'min_p_0.15']:
    candidate = generate_with_strategy(prompt, persona_vector, strategy)
    candidates.append(candidate)
  
  # Apply chain-of-thought editing to each candidate
  edited_candidates = []
  for candidate in candidates:
    edited = apply_cot_editing(candidate, task_type)
    edited_candidates.append(edited)
  
  # Score all candidates (original + edited)
  all_candidates = candidates + edited_candidates
  scores = [wqrm_score(candidate, persona_vector, task_type) 
      for candidate in all_candidates]
  
  # Return highest-scoring output
  best_idx = np.argmax(scores)
  return all_candidates[best_idx], scores[best_idx]
```

## 9. References

1. **Gemma 3 Technical Report**: Core technical insights for parameter-efficient fine-tuning via LoRA targeting MLP and attention layers with 8-bit quantization and optimized learning rates (2e-4).

2. **AI-Slop to AI-Polish? Aligning Language Models through Edit-Based Writing Rewards and Test-time Computation** (Gao et al.): Edit-distance based reward modeling (WQRM), test-time computation pipeline for iterative improvement, and quality metrics for anti-AI-slop training.

3. **Automatic Prediction of Text Aesthetics and Interestingness** (Ganguly et al.): Mathematical formulas for aesthetic scoring including word repetition penalty (1 - unique_words/total_words), average word length, topic diversity via LDA, POS richness, sentiment contrast, and semantic distance metrics.

4. **Contrasting Linguistic Patterns in Human and LLM-Generated News Text** (Muñoz-Ortiz et al.): Dependency length optimality (Ω score), syntactic dependency/constituent distributions, emotion distribution analysis, text similarity metrics, and gender bias measurement for detecting AI-generated content.

5. **Creativity Detection in Texts** (Chiru): Nine automatic measures for distinguishing creative from non-creative text using novelty and quality criteria. Includes Type-to-Token Ratio, Word Norms Fraction, Google Similarity Distance, Explicit Semantic Analysis, Named Entity metrics, WordNet Similarity, Coherence measures, and LSA-based creativity assessment achieving 80% accuracy in creative vs conventional text classification.

6. **Modifying Large Language Model Post-Training for Diverse Creative Writing** (Chung et al.): Diversified Direct Preference Optimization (DDPO) and Odds Ratio Preference Optimization (DORPO) approaches that incorporate deviation weighting to promote both output diversity and quality in creative writing tasks. Introduces quality-diversity balance metrics and demonstrates on-par diversity with human-created datasets while maintaining output quality.

7. **LLMs + Persona-Plug = Personalized LLMs** (Qin et al.): PersonaPlugs methodology for dynamic persona conditioning using user documents to generate persona vectors that guide LLM behavior. Demonstrates superior personalization compared to fine-tuning approaches while maintaining base model capabilities. Key insight: runtime persona conditioning via embeddings rather than parameter modification.

8. **Apple Intelligence Foundation Language Models** (Apple): LoRA adapter architecture with swappable specialized adapters for different tasks. Demonstrates parameter-efficient specialization with 42-57M trainable parameters achieving state-of-the-art results. Emphasizes importance of frozen base models with task-specific LoRA layers for maintaining stability while enabling specialization.

9. **Training Compute-Optimal Large Language Models** (Hoffmann et al., Chinchilla): Scaling laws for optimal compute allocation: models should be trained on ~20x more tokens than parameters (80B tokens for 4B parameters). For LoRA fine-tuning with 42-57M parameters, optimal dataset size is 840M-1.14B tokens. Provides guidance for stage-specific data requirements in multi-stage training pipelines.

10. **Min-p Sampling for Creative and Coherent LLM Outputs** (Facchiano): Novel decoding strategy that sets dynamic thresholds based on maximum token probability. Outperforms temperature and top-k sampling for creative writing by maintaining coherence while preserving creative flexibility. Implementation details for context-sensitive probability filtering.

11. **INSTRUCTION-FOLLOWING SPEECH RECOGNITION** (Lai et al.): Listen-Attend-Spell model architecture for free-form text instruction processing in speech recognition. Enables natural language commands mixed with dictation. Trained from scratch on LibriSpeech without requiring pre-trained speech modules or LLMs. Provides framework for instruction-following audio capabilities.

12. **Large Language Models Are Strong Audio-Visual Speech Recognition Learners** (Cappellazzo et al.): Llama-AVSR architecture achieving state-of-the-art WER 0.81% (ASR) and 0.77% (AVSR) with only 42-57M trainable parameters. Uses frozen encoders (Whisper + AV-HuBERT) with LoRA modules. Demonstrates optimal compression rates (K=3-4 audio, K=2 video) and training data requirements (30h minimum, 433h recommended, 1756h optimal).

13. **Online Automatic Speech Recognition with Listen, Attend and Spell Model** (Hsiao et al.): Solutions for online LAS operation including silence modeling and buffering schemes. Addresses asynchronous decoding challenges through explicit silence tokens. Achieves 12% lower latency than DNN-HMM hybrids. Production-scale deployment insights for real-time speech recognition.

14. **A CONDITIONAL TRANSFORMER LANGUAGE MODEL FOR CONTROLLABLE GENERATION** (Keskar et al., CTRL): Control code methodology for fine-grained generation control using domain, task, and content-specific codes. 1.63B parameter model demonstrating controllable text generation through structured conditioning. Provides framework for combining explicit control codes with persona conditioning for enhanced generation control.

15. **Prompting Large Language Models with Speech Recognition Abilities** (Zhang et al.): Integration strategies for speech capabilities in LLMs using audio embeddings and lightweight projector layers. Demonstrates effective multimodal token processing approaches and provides training methodologies for speech-text integration in large language models.
