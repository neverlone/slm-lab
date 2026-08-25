"""
Data curation, Takatsuki 32k BPE Tokenizer training, and binary sharding pipeline.
Streams high-density, multi-turn conversational, metaphor-rich, and educational text.
"""

import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, decoders, processors
from transformers import PreTrainedTokenizerFast

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|pad|>",
    "<|unk|>",
]

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\n'}}"
    "{% endif %}"
)


def extract_text_from_item(item, source_type: str) -> str:
    """Extracts raw text or formatted multi-turn chat from dataset records."""
    if source_type in ("cosmopedia", "fineweb"):
        return item.get("text", "")
    
    elif source_type == "smoltalk":
        messages = item.get("messages", [])
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        return "\n".join(formatted) + "\n"
    
    elif source_type == "hermes":
        conversations = item.get("conversations", [])
        formatted = []
        for turn in conversations:
            role = turn.get("from", "human")
            if role == "human":
                role = "user"
            elif role == "gpt":
                role = "assistant"
            elif role == "system":
                role = "system"
            content = turn.get("value", "")
            formatted.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        return "\n".join(formatted) + "\n"
        
    return ""


def stream_curated_corpus(max_samples_per_source: int = 50000):
    """
    Streams balanced samples across educational, conversational, metaphor, and uncensored reasoning datasets.
    """
    sources = [
        ("HuggingFaceTB/smoltalk", "all", "train", "smoltalk", max_samples_per_source),
        ("HuggingFaceTB/cosmopedia-v2", "default", "train", "cosmopedia", max_samples_per_source),
        ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "fineweb", max_samples_per_source),
        ("teknium/OpenHermes-2.5", "default", "train", "hermes", max_samples_per_source),
    ]

    for dataset_name, subset, split, source_type, limit in sources:
        print(f"Streaming from source: {dataset_name} ({source_type})...")
        try:
            ds = load_dataset(dataset_name, subset, split=split, streaming=True)
            count = 0
            for item in ds:
                text = extract_text_from_item(item, source_type)
                if text and len(text.strip()) > 30:
                    yield text
                    count += 1
                    if count >= limit:
                        break
            print(f"Loaded {count:,} samples from {dataset_name}.")
        except Exception as e:
            print(f"Warning: could not stream {dataset_name}: {e}. Proceeding with remaining sources.")


def train_takatsuki_tokenizer(
    corpus_file: str,
    vocab_size: int = 32768,
    save_dir: str = "data/tokenizer",
) -> PreTrainedTokenizerFast:
    """Trains the official Takatsuki Byte-Level BPE Tokenizer."""
    os.makedirs(save_dir, exist_ok=True)
    
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=True)
    ])
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    
    print(f"Training Takatsuki BPE Tokenizer (Vocab Size: {vocab_size})...")
    tokenizer.train(files=[corpus_file], trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    json_path = os.path.join(save_dir, "tokenizer.json")
    tokenizer.save(json_path)
    
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=json_path,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        chat_template=CHAT_TEMPLATE,
    )
    fast_tokenizer.save_pretrained(save_dir)
    print(f"Takatsuki Tokenizer successfully saved to {save_dir}")
    return fast_tokenizer


def tokenize_and_shard(
    raw_text_file: str,
    tokenizer_dir: str = "data/tokenizer",
    out_dir: str = "data/tokenized",
    val_split_ratio: float = 0.05,
):
    """Encodes text into memory-mapped uint16 binary token arrays."""
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    print(f"Reading raw corpus from {raw_text_file} for binary tokenization...")
    all_tokens = []
    
    with open(raw_text_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Tokenizing corpus"):
            line = line.strip()
            if not line:
                continue
            token_ids = tokenizer.encode(line, add_special_tokens=False)
            if token_ids:
                all_tokens.extend(token_ids)
                all_tokens.append(eos_id)
                
    total_tokens = len(all_tokens)
    print(f"Total tokens generated: {total_tokens:,}")
    
    # Split train / validation
    n_val = int(total_tokens * val_split_ratio)
    n_train = total_tokens - n_val
    
    train_tokens = np.array(all_tokens[:n_train], dtype=np.uint16)
    val_tokens = np.array(all_tokens[n_train:], dtype=np.uint16)
    
    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")
    
    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)
    
    print(f"Saved Train Shard ({len(train_tokens):,} tokens) -> {train_path} ({os.path.getsize(train_path) / 1e6:.1f} MB)")
    print(f"Saved Val Shard ({len(val_tokens):,} tokens) -> {val_path} ({os.path.getsize(val_path) / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Takatsuki Data Preparation Pipeline")
    parser.add_argument("--samples_per_source", type=int, default=25000, help="Number of samples to stream per dataset")
    parser.add_argument("--vocab_size", type=int, default=32768, help="Vocabulary size for BPE tokenizer")
    args = parser.parse_args()
    
    raw_corpus_path = "data/raw/corpus.txt"
    os.makedirs("data/raw", exist_ok=True)
    
    print("[1/3] Streaming and saving raw training corpus...")
    total_docs = 0
    with open(raw_corpus_path, "w", encoding="utf-8") as f:
        for doc in stream_curated_corpus(max_samples_per_source=args.samples_per_source):
            clean_doc = doc.replace("\r", "").strip()
            if clean_doc:
                f.write(clean_doc + "\n\n")
                total_docs += 1
                if total_docs % 5000 == 0:
                    print(f"Ingested {total_docs:,} documents...")
                    
    print(f"Finished corpus creation: {total_docs:,} documents saved to {raw_corpus_path} ({os.path.getsize(raw_corpus_path) / 1e6:.1f} MB).")
    
    print("\n[2/3] Training Takatsuki 32k BPE Tokenizer...")
    train_takatsuki_tokenizer(corpus_file=raw_corpus_path, vocab_size=args.vocab_size)
    
    print("\n[3/3] Tokenizing corpus into memory-mapped binary shards (train.bin & val.bin)...")
    tokenize_and_shard(raw_text_file=raw_corpus_path)
    
    print("\nTAKATSUKI DATA PREPARATION COMPLETE.")


if __name__ == "__main__":
    main()
