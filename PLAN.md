# Technical Implementation Plan: Dual-Track SLM Development (Option 3)

## Architectural Strategy Overview

This roadmap executes a dual-track strategy to maximize learning and conversational power while staying well within the €250 Oracle Cloud credit budget:

1. **Track 1 (150M Pure Scratch):** Build, train, and align a 150M parameter model completely from zero (random weights) on ~3B high-density tokens. This teaches the full end-to-end science of tokenization, pretraining dynamics, loss curves, and SFT.
2. **Track 2 (3.0B Uncensored Conversational Model):** Build a 3.0B parameter model with deep metaphor comprehension, multi-turn memory, and zero censorship, utilizing continued pretraining, conversational SFT, and DPO alignment.
3. **Deployment on `sen-takatsuki` (Always-Free 24GB RAM):** Both models are converted to GGUF (4-bit / 8-bit) and served locally with a unified Web Chat interface featuring persistent memory.

---

## Credit Budget Allocation Breakdown (€250 Total)

| Phase | Compute Shape | Estimated Hours | Estimated Cost | Remaining Credits |
| :--- | :--- | :--- | :--- | :--- |
| **Track 1: 150M Pre-training (Scratch)** | 1x NVIDIA A10 (24GB VRAM) | ~25 - 30 hrs | ~€35 - €42 | ~€210 |
| **Track 1: 150M SFT Chat Alignment** | 1x NVIDIA A10 | ~4 hrs | ~€6 | ~€204 |
| **Track 2: 3.0B Domain Tuning & SFT** | 1x NVIDIA A10 | ~18 - 22 hrs | ~€25 - €30 | ~€175 |
| **Track 2: 3.0B DPO Alignment** | 1x NVIDIA A10 | ~6 hrs | ~€9 | ~€166 |
| **Buffer for Iterations & Experiments** | — | — | — | **~€166 Surplus** |

---

## Detailed Technical Pipeline

### Phase 1: Dataset Pipeline & Tokenizer (Executed on `sen-takatsuki` for €0)
1. **Corpus Acquisition:**
   - **Foundational & Metaphor Corpus:** Cosmopedia v2 + FineWeb-Edu (scores 4 & 5) + Literary Gutenberg sample (~2.0B tokens).
   - **Conversational Corpus:** SmolTalk + UltraChat 200k (~1.0B tokens).
   - **Uncensored Corpus:** Hermes Uncensored + Dolphin subset (~0.5B tokens).
2. **Tokenizer Training:**
   - Train 32,768 vocabulary Byte-Level BPE on the combined corpus.
   - Special tokens: `<|im_start|>`, `<|im_end|>`, `<|pad|>`, `<|unk|>`, `<|endoftext|>`.
3. **Binary Sharding:**
   - Pre-tokenize the raw stream into `data/tokenized/train.bin` (uint16 / int32 `np.memmap` shards) for instant zero-copy loading on the GPU.

---

### Phase 2: Track 1 Execution (150M From Scratch)
1. **Architecture Specs (150M):**
   - Hidden Dimension: `768`
   - Layers: `12`
   - Attention Heads: `12`
   - KV Heads (GQA): `4`
   - Context Window: `2048` tokens
   - Feed-Forward: SwiGLU (`hidden_dim = 2048`)
   - Normalization: RMSNorm (`eps = 1e-5`)
   - Positional: RoPE (`theta = 10000.0`)
2. **Training Workflow:**
   - Launch on-demand OCI NVIDIA A10 instance.
   - Transfer `train.bin` + tokenizer.
   - Train with Cosine Annealing LR (`6e-4` to `6e-5`), AdamW, BF16 mixed precision.
   - Run SFT chat tuning with conversation masks.
   - Sync checkpoints to `sen-takatsuki` & terminate GPU instance.

---

### Phase 3: Track 2 Execution (3.0B Conversational Powerhouse)
1. **Architecture Specs (3.0B):**
   - Hidden Dimension: `3072`
   - Layers: `28`
   - Attention Heads: `24`
   - KV Heads (GQA): `8`
   - Context Window: `4096` tokens
2. **Alignment & Uncensored Tuning:**
   - Continued pretraining on metaphor and domain text.
   - SFT on multi-turn dialogue with strict removal of refusal templates.
   - Direct Preference Optimization (DPO) for response quality and personality.
   - Sync checkpoints to `sen-takatsuki` & terminate GPU instance.

---

### Phase 4: Quantization, Serving & Web UI on `sen-takatsuki`
1. **GGUF Quantization:**
   - 150M Model: Exported as `150m-q8_0.gguf` (~160 MB RAM, ~60+ tok/s).
   - 3.0B Model: Exported as `3b-q4_k_m.gguf` (~2.2 GB RAM, ~14-18 tok/s).
2. **Web Chat Application (`/home/ubuntu/projects/slm-lab/web`):**
   - FastAPI backend with streaming Server-Sent Events (SSE) / WebSockets.
   - Model switcher in the UI (toggle between 150M and 3B).
   - Local SQLite conversation storage for persistent chat history.
   - System memory injection (stores user facts, preferences, and context).

---

## Verification Plan

### Automated Tests
1. **Data Integrity:** Verify binary token shard boundary alignments, EOS/BOS token placements, and token distribution.
2. **Model Convergence:** Check that validation loss continuously decreases during the first 1,000 steps.
3. **Inference Latency:** Measure time-to-first-token (TTFT) and throughput (tokens/sec) on the 24GB ARM64 CPU.

### Manual Verification
1. Test multi-turn conversational recall in the Web UI.
2. Test figurative language and metaphor comprehension queries.
3. Confirm absence of generic corporate refusal responses on complex prompts.
