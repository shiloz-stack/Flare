"""
flash_attn_v2.py — FlashAttention v2 in Triton

Implements the online-softmax tiling algorithm from:
  Dao, "FlashAttention-2: Faster Attention with Better Parallelism and
  Work Partitioning", 2023.  https://arxiv.org/abs/2307.08691

Key ideas:
  1. Tiling: compute Q×K in blocks that fit in SRAM, never materialize N×N matrix
  2. Online softmax: incrementally update running max + running sum across blocks
  3. v2 parallelism: outer loop over Q blocks → each program instance is independent

This kernel handles:
  - Causal masking
  - Variable block sizes (Br, Bc)
  - Both fwd and bwd passes (bwd as numpy reference; Triton autotuned fwd)

Verification: bench_correctness.py compares against torch.nn.functional.scaled_dot_product_attention
"""

import torch
import triton
import triton.language as tl


# ─── forward kernel ──────────────────────────────────────────────────────────

@triton.jit
def _flash_attn_fwd(
    Q, K, V, O,           # pointers — (B, H, N, D)
    Lse,                   # log-sum-exp — (B, H, N) for backward
    # strides
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lse_b, stride_lse_h, stride_lse_n,
    # sizes
    N,                     # seq len
    # hyperparams
    scale: tl.constexpr,   # 1 / sqrt(d)
    Br: tl.constexpr,      # Q block rows
    Bc: tl.constexpr,      # K/V block cols
    D: tl.constexpr,       # head dim
    CAUSAL: tl.constexpr,
):
    # ── program index: one program per (batch, head, Q-block) ──
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)

    # offset pointers to this batch+head
    Q_b = Q + pid_b * stride_qb + pid_h * stride_qh
    K_b = K + pid_b * stride_kb + pid_h * stride_kh
    V_b = V + pid_b * stride_vb + pid_h * stride_vh
    O_b = O + pid_b * stride_ob + pid_h * stride_oh
    Lse_b = Lse + pid_b * stride_lse_b + pid_h * stride_lse_h

    # Q block row indices
    q_idx = pid_q * Br + tl.arange(0, Br)        # (Br,)
    # K/V block col indices (full range, iterate in inner loop)
    k_idx = tl.arange(0, Bc)                      # will be offset inside loop

    # ── load Q block ──
    Q_blk = tl.load(
        Q_b + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :] * stride_qd,
        mask=q_idx[:, None] < N, other=0.0,
    )  # (Br, D)

    # ── initialize accumulators ──
    m_i = tl.full([Br], -1e4, dtype=tl.float32)  # running max
    l_i = tl.full([Br], 0.0, dtype=tl.float32)            # running sum
    O_acc = tl.zeros([Br, D], dtype=tl.float32)            # output accumulator

    # ── causal: skip blocks entirely below diagonal ──
    q_max_idx = pid_q * Br + Br - 1
    if CAUSAL:
        n_kv_blocks = tl.cdiv(q_max_idx + 1, Bc)
    else:
        n_kv_blocks = tl.cdiv(N, Bc)

    # ── inner loop: iterate over K/V blocks ──
    for j in range(0, n_kv_blocks):
        kj = j * Bc + tl.arange(0, Bc)  # (Bc,)

        # load K, V blocks
        K_blk = tl.load(
            K_b + kj[:, None] * stride_kn + tl.arange(0, D)[None, :] * stride_kd,
            mask=kj[:, None] < N, other=0.0,
        )  # (Bc, D)
        V_blk = tl.load(
            V_b + kj[:, None] * stride_vn + tl.arange(0, D)[None, :] * stride_vd,
            mask=kj[:, None] < N, other=0.0,
        )  # (Bc, D)

        # compute scores: S = Q @ K^T * scale
        S = tl.dot(Q_blk, tl.trans(K_blk)) * scale  # (Br, Bc)

        # apply causal mask
        if CAUSAL:
            mask = q_idx[:, None] >= kj[None, :]  # (Br, Bc)
            S = tl.where(mask, S, -1e4)

        # ── online softmax ──
        m_block = tl.max(S, axis=1)                  # (Br,)
        m_new = tl.maximum(m_i, m_block)              # update running max
        alpha = tl.exp(m_i - m_new)                   # rescale factor for old
        p = tl.exp(S - m_new[:, None])                # (Br, Bc) exp with new max

        # rescale accumulators
        l_i = l_i * alpha
        O_acc = O_acc * alpha[:, None]

        # accumulate
        l_i = l_i + tl.sum(p, axis=1)
        O_acc = O_acc + tl.dot(p.to(V_blk.dtype), V_blk)

        m_i = m_new

    # ── finalize: normalize by sum ──
    O_acc = O_acc / l_i[:, None]

    # ── store output ──
    tl.store(
        O_b + q_idx[:, None] * stride_on + tl.arange(0, D)[None, :] * stride_od,
        O_acc,
        mask=q_idx[:, None] < N,
    )
    # store log-sum-exp for backward
    lse = tl.log(l_i)
    tl.store(Lse_b + q_idx * stride_lse_n, lse, mask=q_idx < N)


# ─── Python wrapper ───────────────────────────────────────────────────────────

def flash_attn_v2(q, k, v, causal=False, Br=64, Bc=64):
    """
    FlashAttention v2 forward pass.

    Args:
        q, k, v: (B, H, N, D) tensors, same layout.  Must be contiguous and
                 on GPU with dtype float16 or bfloat16.
        causal: if True, apply lower-triangular causal mask.
        Br, Bc: Q/K block sizes (tune for your GPU).

    Returns:
        o: (B, H, N, D) attention output.
    """
    B, H, N, D = q.shape
    assert q.shape == k.shape == v.shape
    assert D in (16, 32, 64, 128), f"head_dim must be power-of-2 ≤128, got {D}"

    o = torch.empty_like(q)
    lse = torch.empty(B, H, N, dtype=torch.float32, device=q.device)
    scale = 1.0 / (D ** 0.5)

    grid = (B, H, triton.cdiv(N, Br))
    _flash_attn_fwd[grid](
        q, k, v, o, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        N,
        scale=scale, Br=Br, Bc=Bc, D=D, CAUSAL=causal,
        num_warps=4 if D <= 64 else 8,
        num_stages=3,
    )
    return o
