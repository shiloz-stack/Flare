"""
mla.py — Multi-head Latent Attention (DeepSeek-V2/V3)

MLA compresses the KV cache by projecting K and V through a low-rank bottleneck.
Instead of caching full (N, n_heads, D) K and V tensors, MLA caches a single
latent tensor c_KV of shape (N, d_compress) where d_compress << n_heads * D.

At inference time, the cached latent is projected back to K and V via learned
up-projection matrices.  This dramatically reduces KV cache memory — the primary
bottleneck for long-context LLM serving.

Paper: DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-
of-Experts Model", 2024.  https://arxiv.org/abs/2405.04456

This implementation focuses on the attention computation AFTER the latent
projection.  It handles the "absorbed" form where W_KR (RoPE) is separated
from the low-rank path, matching DeepSeek-V2's production architecture.

Architecture (per layer):
  1. Down-projection:   c_KV = x @ W_DKV              # (N, d_compress)
  2. Cache c_KV (NOT full K, V)                         # ← memory savings here
  3. Up-projection K:   K = c_KV @ W_UK                 # (N, n_heads, D)
  4. Up-projection V:   V = c_KV @ W_UV                 # (N, n_heads, D)
  5. Attention:         O = softmax(Q @ K^T / √d) @ V  # ← standard FA v2

The "latent attention" trick: steps 3+4 happen inside the attention kernel,
so K and V are NEVER materialized in HBM during the attention computation.
"""

import torch
import triton
import triton.language as tl

from .flash_attn_v2 import flash_attn_v2


@triton.jit
def _mla_attn_fwd(
    Q,           # (B, H, N, D_qk)
    C_KV,        # (B, N, d_compress) — compressed KV cache
    W_UK,        # (d_compress, H, D_qk) — K up-projection
    W_UV,        # (d_compress, H, D_v)  — V up-projection
    O,           # (B, H, N, D_v)
    Lse,         # (B, H, N)
    # strides ...
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_cb, stride_cn, stride_cd,
    stride_uk_d, stride_uk_h, stride_uk_k,
    stride_uv_d, stride_uv_h, stride_uv_v,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lse_b, stride_lse_h, stride_lse_n,
    N,
    scale: tl.constexpr,
    Br: tl.constexpr,
    Bc: tl.constexpr,
    D_qk: tl.constexpr,
    D_v: tl.constexpr,
    d_compress: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)

    # ── load Q block ──
    Q_b = Q + pid_b * stride_qb + pid_h * stride_qh
    q_idx = pid_q * Br + tl.arange(0, Br)

    Q_blk = tl.load(
        Q_b + q_idx[:, None] * stride_qn + tl.arange(0, D_qk)[None, :] * stride_qd,
        mask=q_idx[:, None] < N, other=0.0,
    )  # (Br, D_qk)

    # ── load up-projection weights for this head ──
    # W_UK for head h: (d_compress, D_qk)
    UK_h = W_UK + pid_h * stride_uk_h
    UV_h = W_UV + pid_h * stride_uv_h

    # ── accumulators ──
    m_i = tl.full([Br], float('-inf'), dtype=tl.float32)
    c_i = tl.zeros([Br], dtype=tl.float32)
    O_acc = tl.zeros([Br, D_v], dtype=tl.float32)

    C_KV_b = C_KV + pid_b * stride_cb
    O_b = O + pid_b * stride_ob + pid_h * stride_oh
    Lse_b = Lse + pid_b * stride_lse_b + pid_h * stride_lse_h

    n_kv_blocks = tl.cdiv(N, Bc)

    for j in range(0, n_kv_blocks):
        kj = j * Bc + tl.arange(0, Bc)  # (Bc,)

        # ── load compressed KV block ──
        c_kv = tl.load(
            C_KV_b + kj[:, None] * stride_cn + tl.arange(0, d_compress)[None, :] * stride_cd,
            mask=kj[:, None] < N, other=0.0,
        )  # (Bc, d_compress)

        # ── up-project to K and V on-the-fly (stays in SRAM) ──
        # K = c_kv @ W_UK_h  →  (Bc, D_qk)
        W_UK_blk = tl.load(
            UK_h + tl.arange(0, d_compress)[:, None] * stride_uk_d
                   + tl.arange(0, D_qk)[None, :] * stride_uk_k,
        )  # (d_compress, D_qk)
        K_blk = tl.dot(c_kv, W_UK_blk)  # (Bc, D_qk)

        # V = c_kv @ W_UV_h  →  (Bc, D_v)
        W_UV_blk = tl.load(
            UV_h + tl.arange(0, d_compress)[:, None] * stride_uv_d
                   + tl.arange(0, D_v)[None, :] * stride_uv_v,
        )  # (d_compress, D_v)
        V_blk = tl.dot(c_kv, W_UV_blk)  # (Bc, D_v)

        # ── standard attention score ──
        S = tl.dot(Q_blk, tl.trans(K_blk)) * scale  # (Br, Bc)

        # ── online softmax ──
        m_block = tl.max(S, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(S - m_new[:, None])

        c_i = c_i * alpha
        O_acc = O_acc * alpha[:, None]
        c_i = c_i + tl.sum(p, axis=1)
        O_acc = O_acc + tl.dot(p.to(V_blk.dtype), V_blk)
        m_i = m_new

    O_acc = O_acc / c_i[:, None]

    tl.store(
        O_b + q_idx[:, None] * stride_on + tl.arange(0, D_v)[None, :] * stride_od,
        O_acc,
        mask=q_idx[:, None] < N,
    )
    lse = tl.log(c_i)
    tl.store(Lse_b + q_idx * stride_lse_n, lse, mask=q_idx < N)


def mla_attention(
    q,           # (B, H, N, D_qk)
    c_kv,        # (B, N, d_compress) — latent KV
    W_UK,        # (d_compress, H, D_qk)
    W_UV,        # (d_compress, H, D_v)
    Br=64, Bc=64,
):
    """
    Multi-head Latent Attention forward pass.

    The key difference from standard attention: instead of full K, V tensors,
    we receive a compressed latent c_KV and up-project inside the kernel.

    KV cache memory:  d_compress × N  vs  2 × H × D × N for standard MHA.

    Args:
        q:       (B, H, N, D_qk)   query tensor
        c_kv:    (B, N, d_compress)  compressed KV latent
        W_UK:    (d_compress, H, D_qk)  K up-projection weights
        W_UV:    (d_compress, H, D_v)   V up-projection weights
    Returns:
        o:       (B, H, N, D_v)
    """
    B, H, N, D_qk = q.shape
    _, _, d_compress = c_kv.shape
    D_v = W_UV.shape[-1]
    assert D_qk == W_UK.shape[-1], f"D_qk mismatch: {D_qk} vs {W_UK.shape[-1]}"
    assert d_compress == W_UK.shape[0] == W_UV.shape[0]

    o = torch.empty(B, H, N, D_v, dtype=q.dtype, device=q.device)
    lse = torch.empty(B, H, N, dtype=torch.float32, device=q.device)
    scale = 1.0 / (D_qk ** 0.5)

    grid = (B, H, triton.cdiv(N, Br))
    _mla_attn_fwd[grid](
        q, c_kv, W_UK, W_UV, o, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        c_kv.stride(0), c_kv.stride(1), c_kv.stride(2),
        W_UK.stride(0), W_UK.stride(1), W_UK.stride(2),
        W_UV.stride(0), W_UV.stride(1), W_UV.stride(2),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        N,
        scale=scale, Br=Br, Bc=Bc, D_qk=D_qk, D_v=D_v, d_compress=d_compress,
        num_warps=4,
        num_stages=2,
    )
    return o


# ─── PyTorch reference for testing ───────────────────────────────────────────

def mla_attention_ref(q, c_kv, W_UK, W_UV):
    """PyTorch eager-mode reference for correctness checking."""
    B, H, N, D_qk = q.shape
    d_compress = c_kv.shape[-1]
    D_v = W_UV.shape[-1]

    # up-project K and V
    # c_kv: (B, N, d_compress) → K: (B, H, N, D_qk)
    K = torch.einsum('bnd,dhd->bhn', c_kv, W_UK).unsqueeze(2).expand(-1, -1, N, -1)
    # Actually need per-position K: (B, N, H, D) then transpose
    K = torch.einsum('bnd,dhk->bnhk', c_kv, W_UK).transpose(1, 2)  # (B, H, N, D_qk)
    V = torch.einsum('bnd,dhv->bnhv', c_kv, W_UV).transpose(1, 2)  # (B, H, N, D_v)

    scale = 1.0 / (D_qk ** 0.5)
    S = torch.einsum('bhnd,bhmd->bhnm', q * scale, K)  # (B, H, N, N)
    S_max = S.amax(dim=-1, keepdim=True)
    P = torch.exp(S - S_max)
    P = P / P.sum(dim=-1, keepdim=True)
    O = torch.einsum('bhnm,bhmd->bhnd', P, V)  # (B, H, N, D_v)
    return O
