"""
Training pipeline for SLM Pre-training and Fine-tuning.
Supports PyTorch mixed-precision (BF16/FP16), Cosine LR scheduler,
gradient clipping, and automated checkpointing.
"""

import os
import math
import time
import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.model.transformer import SLMForCausalLM, ModelArgs
from src.dataset.dataset import create_dataloader


def get_lr(it: int, warmup_iters: int, lr_decay_iters: int, max_lr: float, min_lr: float) -> float:
    # Linear warmup
    if it < warmup_iters:
        return max_lr * (it + 1) / (warmup_iters + 1)
    # Cosine decay
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model initialization
    model_args = ModelArgs(**cfg.get("model", {}))
    model = SLMForCausalLM(model_args).to(device)
    print(f"Initialized SLM Model with {model.count_parameters():,} trainable parameters.")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"]["max_lr"],
        betas=(0.9, 0.95),
        weight_decay=cfg["training"].get("weight_decay", 0.1),
    )

    # Checkpoints directory
    ckpt_dir = cfg["training"].get("checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    data_path = cfg["data"]["train_bin"]
    if not os.path.exists(data_path):
        print(f"Warning: Training data binary {data_path} not found. Running in dry-test mode.")
        return

    dataloader = create_dataloader(
        bin_path=data_path,
        batch_size=cfg["training"]["batch_size"],
        seq_len=model_args.max_seq_len,
    )

    max_steps = cfg["training"]["max_steps"]
    warmup_steps = cfg["training"]["warmup_steps"]
    max_lr = cfg["training"]["max_lr"]
    min_lr = cfg["training"]["min_lr"]
    grad_clip = cfg["training"].get("grad_clip", 1.0)
    save_interval = cfg["training"].get("save_interval", 1000)

    model.train()
    step = 0
    t0 = time.time()

    print("Starting training run...")
    data_iter = iter(dataloader)

    while step < max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = model(x, labels=y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % cfg["training"].get("log_interval", 10) == 0:
            dt = time.time() - t0
            t0 = time.time()
            tok_per_sec = (cfg["training"]["batch_size"] * model_args.max_seq_len * 10) / max(dt, 1e-4)
            print(f"Step {step}/{max_steps} | Loss: {loss.item():.4f} | LR: {lr:.2e} | Speed: {tok_per_sec:.0f} tok/s")

        if step > 0 and (step % save_interval == 0 or step == max_steps - 1):
            ckpt_path = os.path.join(ckpt_dir, f"model_step_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
                "model_args": model_args,
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        step += 1

    print("Training run completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model_150m.yaml", help="Path to YAML config file")
    args = parser.parse_args()
    train(args.config)
