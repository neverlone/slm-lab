"""
Unit test suite for SLM Tokenizer, Transformer architecture, and generation.
"""

import os
import torch
from src.tokenizer.tokenizer import train_tokenizer, load_tokenizer, SPECIAL_TOKENS
from src.model.transformer import SLMForCausalLM, ModelArgs


def test_tokenizer():
    print("Testing Tokenizer...")
    sample_texts = [
        "Hello world! This is a test for our custom Small Language Model.",
        "The quick brown fox jumps over the lazy dog.",
        "<|im_start|>user\nWhat is an SLM?<|im_end|>\n<|im_start|>assistant\nAn SLM is a Small Language Model.<|im_end|>",
    ]
    test_dir = "/tmp/test_tokenizer"
    train_tokenizer(sample_texts, vocab_size=512, save_directory=test_dir, min_frequency=1)
    
    tokenizer = load_tokenizer(test_dir)
    encoded = tokenizer.encode("<|im_start|>user\nHello!<|im_end|>")
    decoded = tokenizer.decode(encoded)
    
    assert "<|im_start|>" in decoded, "Special token <|im_start|> missing from decoded text"
    assert "<|im_end|>" in decoded, "Special token <|im_end|> missing from decoded text"
    print("✓ Tokenizer test passed.")


def test_model_forward_and_loss():
    print("Testing Transformer Architecture & Forward Pass...")
    args = ModelArgs(
        dim=256,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=1024,
        max_seq_len=128,
    )
    model = SLMForCausalLM(args)
    print(f"Mini Test Model Parameters: {model.count_parameters():,}")
    
    # Forward pass without labels
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, args.vocab_size, (batch_size, seq_len))
    logits, loss, _ = model(input_ids)
    
    assert logits.shape == (batch_size, seq_len, args.vocab_size), f"Unexpected logits shape: {logits.shape}"
    assert loss is None, "Loss should be None when labels are not provided"
    
    # Forward pass with labels (cross entropy loss)
    labels = input_ids.clone()
    logits, loss, _ = model(input_ids, labels=labels)
    assert loss is not None and loss.item() > 0, "Loss computation failed"
    print(f"✓ Forward pass & loss test passed (Test Loss: {loss.item():.4f}).")


def test_model_generation():
    print("Testing Autoregressive Generation...")
    args = ModelArgs(
        dim=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=512,
        max_seq_len=64,
    )
    model = SLMForCausalLM(args)
    model.eval()
    
    prompt = torch.tensor([[10, 20, 30]])
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.8)
    assert generated.shape == (1, 13), f"Expected generated shape (1, 13), got {generated.shape}"
    print("✓ Autoregressive generation test passed.")


if __name__ == "__main__":
    test_tokenizer()
    test_model_forward_and_loss()
    test_model_generation()
    print("\nALL ARCHITECTURE TESTS PASSED SUCCESSFULLY.")
