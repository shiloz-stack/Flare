"""
mla.py — Multi-head Latent Attention (DeepSeek-V2/V3)

MLA compresses the KV cache by projecting K and V through a low-rank bottleneck.
Instead of caching full (N, n_heads, D) K and V tensors, MLA caches a single
latent tensor c_KV of shape (N, d_compress) where d_compress << n_heads * D.

At inference time, the cached latent is projected back to K and V via learned
up-projection matrices. This dramatically reduces KV cache memory — the primary
bottleneck for long-context LLM serving.

Paper: DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-
of-Experts Model", 2024.  https://arxiv.org/abs/2405.04456

The "latent attention" trick: up-projection happens inside the attention kernel,
so full K and V are NEVER materialized in HBM during the attention computation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mla_attn_fwd(
    Q,           # (B, H, N, D)
    C_KV,        # (B, N, d_compress) — compressed KV cache
    W_UK,        # (d_compress, H, D) — K up-projection
    W_UV,        # (d_compress, H, D) — V up-projection
    O,           # (B, H, N, D)
    Lse,         # (B, H, N)
    # strides
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
    D: tl.constexpr,
    d_compress: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)

    # ── load Q block ──
    Q_b = Q + pid_b * stride_qb + pid_h * stride_qh
    q_idx = pid_q * Br + tl.arange(0, Br)

    Q_blk = tl.load(
        Q_b + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :] * stride_qd,
        mask=q_idx[:, None] < N, other=0.0,
    )  # (Br, D) fp16

    # ── load up-projection weights ONCE (constant per head) ──
    UK_h = W_UK + pid_h * stride_uk_h
    UV_h = W_UV + pid_h * stride_uv_h

    W_UK_blk = tl.load(
        UK_h + tl.arange(0, d_compress)[:, None] * stride_uk_d
               + tl.arange(0, D)[None, :] * stride_uk_k,
    )  # (d_compress, D) fp16

    W_UV_blk = tl.load(
        UV_h + tl.arange(0, d_compress)[:, None] * stride_uv_d
               + tl.arange(0, D)[None, :] * stride_uv_v,
    )  # (d_compress, D) fp16

    # ── accumulators (fp32) ──
    m_i = tl.full([Br], -1e4, dtype=tl.float32)
    l_i = tl.full([Br], 0.0, dtype=tl.float32)
    O_acc = tl.zeros([Br, D], dtype=tl.float32)

    C_KV_b = C_KV + pid_b * stride_cb
    O_b = O + pid_b * stride_ob + pid_h * stride_oh
    Lse_b = Lse + pid_b * stride_lse_b + pid_h * stride_lse_h

    n_kv_blocks = tl.cdiv(N, Bc)

    for j in range(0, n_kv_blocks):
        kj = j * Bc + tl.arange(0, Bc)

        # ── load compressed KV block ──
        c_kv = tl.load(
            C_KV_b + kj[:, None] * stride_cn + tl.arange(0, d_compress)[None, :] * stride_cd,
            mask=kj[:, None] < N, other=0.0,
        )  # (Bc, d_compress) fp16

        # ── up-project K and V on-the-fly (fp16 × fp16 → fp32, cast back to fp16) ──
        K_blk = tl.dot(c_kv, W_UK_blk).to(tl.float16)  # (Bc, D) fp16
        V_blk = tl.dot(c_kv, W_UV_blk).to(tl.float16)  # (Bc, D) fp16

        # ── attention scores (fp16 × fp16 → fp32) ──
        S = tl.dot(Q_blk, tl.trans(K_blk)) * scale  # (Br, Bc) fp32

        # ── online softmax (fp32) ──
        m_block = tl.max(S, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(S - m_new[:, None])  # (Br, Bc) fp32

        l_i = l_i * alpha
        O_acc = O_acc * alpha[:, None]
        l_i = l_i + tl.sum(p, axis=1)
        O_acc = O_acc + tl.dot(p.to(tl.float16), V_blk)  # cast p to fp16 for dot

        m_i = m_new

    # ── finalize ──
    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    O_acc = O_acc / l_i_safe[:, None]

    tl.store(
        O_b + q_idx[:, None] * stride_on + tl.arange(0, D)[None, :] * stride_od,
        O_acc.to(tl.float16),
        mask=q_idx[:, None] < N,
    )
    lse = tl.log(l_i_safe)
    tl.store(Lse_b + q_idx * stride_lse_n, lse, mask=q_idx < N)


def mla_attention(
    q,           # (B, H, N, D)
    c_kv,        # (B, N, d_compress)
    W_UK,        # (d_compress, H, D)
    W_UV,        # (d_compress, H, D)
    Br=64, Bc=64,
):
    """
    Multi-head Latent Attention forward pass.

    KV cache memory:  d_compress × N  vs  2 × H × D × N for standard MHA.
    """
    B, H, N, D = q.shape
    _, _, d_compress = c_kv.shape
    assert D == W_UK.shape[-1] == W_UV.shape[-1]
    assert d_compress == W_UK.shape[0] == W_UV.shape[0]

    o = torch.empty(B, H, N, D, dtype=q.dtype, device=q.device)
    lse = torch.empty(B, H, N, dtype=torch.float32, device=q.device)
    scale = 1.0 / (D ** 0.5)

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
        scale=scale, Br=Br, Bc=Bc, D=D, d_compress=d_compress,
        num_warps=4,
        num_stages=2,
    )
    return o


# ─── PyTorch reference for testing ───────────────────────────────────────────

def mla_attention_ref(q, c_kv, W_UK, W_UV):
    """PyTorch reference — computes in fp32 for maximum precision."""
    B, H, N, D = q.shape
    d_compress = c_kv.shape[-1]

    # Compute in fp32 to serve as ground truth
    q_f32 = q.float()
    c_kv_f32 = c_kv.float()
    W_UK_f32 = W_UK.float()
    W_UV_f32 = W_UV.float()

    # up-project K and V
    K = torch.einsum('bnd,dhk->bnhk', c_kv_f32, W_UK_f32).transpose(1, 2)  # (B, H, N, D)
    V = torch.einsum('bnd,dhv->bnhv', c_kv_f32, W_UV_f32).transpose(1, 2)  # (B, H, N, D)

    scale = 1.0 / (D ** 0.5)
    S = torch.einsum('bhnd,bhmd->bhnm', q_f32 * scale, K)  # (B, H, N, N)
    S_max = S.amax(dim=-1, keepdim=True)
    P = torch.exp(S - S_max)
    P = P / P.sum(dim=-1, keepdim=True)
    O = torch.einsum('bhnm,bhmd->bhnd', P, V)  # (B, H, N, D)
    return O.to(q.dtype)
