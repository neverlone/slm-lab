"""
Clean, modern PyTorch Transformer Decoder architecture (Llama-3 / SmolLM design).
Features:
- Pre-RMSNorm
- Rotary Position Embeddings (RoPE)
- SwiGLU Feed-Forward Network
- Grouped Query Attention (GQA) with PyTorch SDPA (FlashAttention kernel dispatch)
- KV-Cache support for fast autoregressive generation
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 768
    n_layers: int = 16
    n_heads: int = 12
    n_kv_heads: Optional[int] = 4  # GQA (3 query heads per KV head)
    vocab_size: int = 32768
    hidden_dim: Optional[int] = None
    multiple_of: int = 256
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.hidden_dim is None:
            # SwiGLU 8/3 hidden dim calculation aligned to multiple_of
            hidden_dim = int(2 * (4 * self.dim) / 3)
            self.hidden_dim = self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq: [B, H, S, D], xk: [B, H_kv, S, D]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, 1, xq_.shape[2], xq_.shape[3])
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats KV heads to match query head count in GQA: [B, n_kv_heads, S, D] -> [B, n_heads, S, D]"""
    if n_rep == 1:
        return x
    bs, n_kv_heads, slen, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


class GroupedQueryAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seqlen, _ = x.shape
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            xk = torch.cat([k_cache, xk], dim=2)
            xv = torch.cat([v_cache, xv], dim=2)
            new_kv_cache = (xk, xv)
        else:
            new_kv_cache = None

        # Repeat KV heads for GQA
        key = repeat_kv(xk, self.n_rep)
        value = repeat_kv(xv, self.n_rep)

        # PyTorch native scaled dot-product attention (dispatches FlashAttention / MemEfficient / SDPA)
        is_causal = mask is None and seqlen > 1 and kv_cache is None
        output = F.scaled_dot_product_attention(
            xq, key, value, attn_mask=mask, is_causal=is_causal
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output), new_kv_cache


class SwiGLUMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)  # Gate
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)  # Down
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)  # Up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = GroupedQueryAttention(args)
        self.feed_forward = SwiGLUMLP(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        h, new_kv = self.attention(self.attention_norm(x), freqs_cis, mask=mask, kv_cache=kv_cache)
        x = x + h
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_kv


class SLMForCausalLM(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        if args.tie_word_embeddings:
            self.output.weight = self.tok_embeddings.weight

        # Precompute RoPE complex frequencies buffer
        freqs_cis = precompute_freqs_cis(
            args.dim // args.n_heads,
            args.max_seq_len * 2,
            args.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        kv_caches: Optional[list] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        bsz, seqlen = input_ids.shape
        h = self.tok_embeddings(input_ids)
        
        freqs_cis = self.freqs_cis[:seqlen]
        new_kv_caches = [] if kv_caches is not None else None

        for idx, layer in enumerate(self.layers):
            layer_cache = kv_caches[idx] if kv_caches is not None else None
            h, new_cache = layer(h, freqs_cis, kv_cache=layer_cache)
            if new_kv_caches is not None:
                new_kv_caches.append(new_cache)

        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss, new_kv_caches

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 50,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= self.args.max_seq_len else input_ids[:, -self.args.max_seq_len:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)

            if eos_id is not None and (next_token == eos_id).all():
                break

        return input_ids
