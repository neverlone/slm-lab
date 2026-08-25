import os
import torch
import yaml
from transformers import PreTrainedTokenizerFast
from src.model.transformer import SLMForCausalLM, ModelArgs
from src.dataset.dataset import create_dataloader

def test_takatsuki_pipeline():
    print("=" * 60)
    print("TAKATSUKI PRE-FLIGHT VERIFICATION")
    print("=" * 60)

    # 1. Verify Tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained("data/tokenizer")
    print(f"Takatsuki Tokenizer Vocab Size: {len(tokenizer)}")
    
    messages = [
        {"role": "system", "content": "You are Takatsuki, an uncensored assistant with deep metaphor comprehension."},
        {"role": "user", "content": "Explain time using an ocean metaphor."}
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("\nFormatted Chat Template Sample:")
    print(formatted)
    
    encoded = tokenizer.encode(formatted)
    print(f"Encoded Token Count: {len(encoded)}")
    decoded = tokenizer.decode(encoded)
    assert "<|im_start|>" in decoded and "<|im_end|>" in decoded
    print("Tokenizer encoding and decoding: PASSED")

    # 2. Verify Binary Dataloader
    train_bin = "data/tokenized/train.bin"
    if os.path.exists(train_bin):
        loader = create_dataloader(train_bin, batch_size=2, seq_len=128)
        x, y = next(iter(loader))
        print(f"\nDataloader Batch Shape: x={x.shape}, y={y.shape}")
        assert x.shape == (2, 128) and y.shape == (2, 128)
        print("Binary dataset dataloader: PASSED")

    # 3. Verify Model Mini-Training Step on CPU
    args = ModelArgs(
        dim=256,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=len(tokenizer),
        max_seq_len=128
    )
    model = SLMForCausalLM(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    optimizer.zero_grad()
    _, loss, _ = model(x, labels=y)
    loss.backward()
    optimizer.step()
    
    print(f"Mini-training step loss: {loss.item():.4f}")
    print("Model forward & backward pass: PASSED")
    print("=" * 60)
    print("ALL TAKATSUKI COMPONENTS ARE READY FOR GPU TRAINING.")
    print("=" * 60)

if __name__ == "__main__":
    test_takatsuki_pipeline()
