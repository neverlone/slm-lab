"""
Takatsuki-150M Pre-training Engine.
Optimized for multi-threaded CPU execution with Cosine Annealing, AdamW,
gradient clipping, and live progress logging.
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
from src.dataset.dataset import create_multi_shard_dataloader


def get_lr(it: int, warmup_iters: int, lr_decay_iters: int, max_lr: float, min_lr: float) -> float:
    if it < warmup_iters:
        return max_lr * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model_150m.yaml", help="Path to config")
    parser.add_argument("--shards_dir", type=str, default="data/tokenized_shards", help="Shards folder")
    parser.add_argument("--threads", type=int, default=16, help="CPU threads to use")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    print(f"Set PyTorch CPU threads to {args.threads}")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}")

    model_args = ModelArgs(**cfg.get("model", {}))
    model = SLMForCausalLM(model_args).to(device)
    print(f"Initialized Takatsuki-150M with {model.count_parameters():,} trainable parameters.")

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["training"]["max_lr"]),
        betas=(0.9, 0.95),
        weight_decay=float(cfg["training"].get("weight_decay", 0.1)),
    )

    ckpt_dir = cfg["training"].get("checkpoint_dir", "checkpoints/takatsuki_150m")
    os.makedirs(ckpt_dir, exist_ok=True)

    dataloader = create_multi_shard_dataloader(
        shards_dir=args.shards_dir,
        batch_size=cfg["training"].get("batch_size", 4),
        seq_len=model_args.max_seq_len,
        num_workers=2,
    )

    max_steps = int(cfg["training"].get("max_steps", 25000))
    warmup_steps = int(cfg["training"].get("warmup_steps", 500))
    max_lr = float(cfg["training"].get("max_lr", 6.0e-4))
    min_lr = float(cfg["training"].get("min_lr", 6.0e-5))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))
    save_interval = int(cfg["training"].get("save_interval", 500))
    log_interval = int(cfg["training"].get("log_interval", 5))

    model.train()
    step = 0
    t0 = time.time()
    batch_size = cfg["training"].get("batch_size", 4)

    print("\n" + "=" * 60)
    print(f"   🚀 TAKATSUKI-150M PRE-TRAINING RUN ACTIVE ({max_steps:,} steps)   ")
    print("=" * 60)

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

        if step % log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            tok_processed = batch_size * model_args.max_seq_len * log_interval
            tok_per_sec = tok_processed / max(dt, 1e-4)
            pct = (step / max_steps) * 100
            print(f"[Step {step:05d}/{max_steps:05d} ({pct:4.1f}%)] Loss: {loss.item():.4f} | LR: {lr:.2e} | Speed: {tok_per_sec:,.0f} tok/s | Elapsed: {dt:.2f}s")

        if step > 0 and (step % save_interval == 0 or step == max_steps - 1):
            ckpt_path = os.path.join(ckpt_dir, f"takatsuki_150m_step_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
                "model_args": model_args,
            }, ckpt_path)
            print(f"--> Saved Takatsuki Checkpoint: {ckpt_path} (Loss: {loss.item():.4f})")

        step += 1

    print("\n✅ TAKATSUKI-150M TRAINING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    train()
