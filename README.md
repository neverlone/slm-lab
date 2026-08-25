# SLM Lab

A modular framework for designing, tokenizing, pre-training, instruction fine-tuning (SFT), and deploying Small Language Models (SLMs).

## Architecture Overview

- Model Type: Decoder-only Transformer
- Normalization: RMSNorm (Pre-normalization)
- Positional Embeddings: Rotary Position Embeddings (RoPE)
- Feed-Forward Network: SwiGLU Activation
- Attention Mechanism: Grouped-Query Attention (GQA) with FlashAttention support
- Tokenizer: Custom Byte-Pair Encoding (BPE) with dedicated chat tokens

## Project Structure

- \configs/\: Model architecture and training hyperparameter configurations.
- \src/\:
  - \model/\: PyTorch Transformer architecture implementation.
  - \	okenizer/\: Tokenizer training, encoding, and vocab management.
  - \dataset/\: Dataloaders and memory-mapped binary dataset streaming.
  - \	raining/\: Pre-training, instruction tuning (SFT), and alignment routines.
  - \export/\: Model serialization, Hugging Face conversion, and GGUF quantization.
- \web/\: Inference server and web interface.
- \scripts/\: Dataset preparation and deployment utilities.

## Requirements

- Python >= 3.10
- PyTorch >= 2.1
- uv (package management)

## Getting Started

1. Set up the virtual environment:
   \\\ash
   uv venv
   source .venv/bin/activate
   \\\

2. Install dependencies:
   \\\ash
   uv pip install -r requirements.txt
   \\\
