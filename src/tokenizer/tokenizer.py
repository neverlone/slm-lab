"""
Custom Byte-Pair Encoding (BPE) Tokenizer implementation.
Uses Hugging Face tokenizers library with ByteLevel pre-tokenization
and dedicated special tokens for conversational SLMs.
"""

import os
from typing import List, Optional, Union, Dict
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, decoders, processors
from transformers import PreTrainedTokenizerFast


SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|pad|>",
    "<|unk|>",
]


def create_bpe_tokenizer(vocab_size: int = 32768) -> Tokenizer:
    """
    Initializes a Byte-Level BPE Tokenizer with standard GPT-4/LLaMA splitting patterns.
    """
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    
    # Pre-tokenization: ByteLevel splits whitespace while preserving all raw bytes
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=True)
    ])
    
    # Decoder: ByteLevel reconstruction
    tokenizer.decoder = decoders.ByteLevel()
    
    return tokenizer


def train_tokenizer(
    text_iterator,
    vocab_size: int = 32768,
    save_directory: str = "data/tokenizer",
    min_frequency: int = 2,
) -> str:
    """
    Trains BPE tokenizer over an iterator of texts and saves configuration.
    """
    os.makedirs(save_directory, exist_ok=True)
    tokenizer = create_bpe_tokenizer(vocab_size=vocab_size)
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    
    print(f"Training BPE tokenizer (Target Vocab Size: {vocab_size})...")
    tokenizer.train_from_iterator(text_iterator, trainer=trainer)
    
    # Post-processor for special tokens
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    json_path = os.path.join(save_directory, "tokenizer.json")
    tokenizer.save(json_path)
    print(f"Saved raw tokenizer to {json_path}")
    
    # Wrap in HuggingFace Fast Tokenizer for seamless interoperability
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=json_path,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
    )
    fast_tokenizer.save_pretrained(save_directory)
    print(f"Saved HuggingFace compatible tokenizer to {save_directory}")
    
    return save_directory


def load_tokenizer(tokenizer_dir: str = "data/tokenizer") -> PreTrainedTokenizerFast:
    """
    Loads saved tokenizer.
    """
    return PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
