"""
sliding_window.py — Sliding-Window Attention (Mistral 7B)

A minimal extension of FlashAttention v2: instead of causal mask
(every token attends to all previous), use a local window mask
(each token only attends to W previous tokens).

Complexity drops from O(N²) to O(N×W), enabling long-context generation
without quadratic memory growth.

Paper: Jiang et al., "Mistral 7B", 2023.  https://arxiv.org/abs/2310.06825

The implementation reuses the FA v2 kernel with a different mask:
  causal:  q_idx >= k_idx
  sliding: q_idx - W < k_idx <= q_idx
"""

import torch
import triton
import triton.language as tl

from .flash_attn_v2 import flash_attn_v2


@triton.jit
def _sliding_window_attn_fwd(
    Q, K, V, O,
    Lse,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lse_b, stride_lse_h, stride_lse_n,
    N,
    scale: tl.constexpr,
    Br: tl.constexpr,
    Bc: tl.constexpr,
    D: tl.constexpr,
    WINDOW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)

    Q_b = Q + pid_b * stride_qb + pid_h * stride_qh
    K_b = K + pid_b * stride_kb + pid_h * stride_kh
    V_b = V + pid_b * stride_vb + pid_h * stride_vh
    O_b = O + pid_b * stride_ob + pid_h * stride_oh
    Lse_b = Lse + pid_b * stride_lse_b + pid_h * stride_lse_h

    q_idx = pid_q * Br + tl.arange(0, Br)
    k_idx_base = tl.arange(0, Bc)

    Q_blk = tl.load(
        Q_b + q_idx[:, None] * stride_qn + tl.arange(0, D)[None, :] * stride_qd,
        mask=q_idx[:, None] < N, other=0.0,
    )

    m_i = tl.full([Br], float('-inf'), dtype=tl.float32)
    l_i = tl.full([Br], 0.0, dtype=tl.float32)
    O_acc = tl.zeros([Br, D], dtype=tl.float32)

    # ── sliding window: only iterate over K/V blocks within [q_idx - WINDOW, q_idx] ──
    # The first relevant K/V block for this Q block:
    q_min_idx = pid_q * Br
    j_start = max(0, (q_min_idx - WINDOW) // Bc)
    q_max_idx = pid_q * Br + Br - 1
    j_end = tl.cdiv(q_max_idx + 1, Bc)

    for j in range(j_start, j_end):
        kj = j * Bc + k_idx_base

        K_blk = tl.load(
            K_b + kj[:, None] * stride_kn + tl.arange(0, D)[None, :] * stride_kd,
            mask=kj[:, None] < N, other=0.0,
        )
        V_blk = tl.load(
            V_b + kj[:, None] * stride_vn + tl.arange(0, D)[None, :] * stride_vd,
            mask=kj[:, None] < N, other=0.0,
        )

        S = tl.dot(Q_blk, tl.trans(K_blk)) * scale

        # ── sliding window + causal mask ──
        # valid: q_idx - WINDOW < k_idx <= q_idx
        mask = (q_idx[:, None] >= kj[None, :]) & \
               (q_idx[:, None] - kj[None, :] < WINDOW)
        S = tl.where(mask, S, float('-inf'))

        m_block = tl.max(S, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(S - m_new[:, None])

        l_i = l_i * alpha
        O_acc = O_acc * alpha[:, None]
        l_i = l_i + tl.sum(p, axis=1)
        O_acc = O_acc + tl.dot(p.to(V_blk.dtype), V_blk)
        m_i = m_new

    O_acc = O_acc / l_i[:, None]

    tl.store(
        O_b + q_idx[:, None] * stride_on + tl.arange(0, D)[None, :] * stride_od,
        O_acc,
        mask=q_idx[:, None] < N,
    )
    lse = tl.log(l_i)
    tl.store(Lse_b + q_idx * stride_lse_n, lse, mask=q_idx < N)


def sliding_window_attn(q, k, v, window=512, Br=64, Bc=64):
    """
    Sliding-window causal attention.

    Args:
        q, k, v: (B, H, N, D)
        window: number of past tokens each position can attend to.
        Br, Bc: block sizes.
    Returns:
        o: (B, H, N, D)
    """
    B, H, N, D = q.shape
    o = torch.empty_like(q)
    lse = torch.empty(B, H, N, dtype=torch.float32, device=q.device)
    scale = 1.0 / (D ** 0.5)

    grid = (B, H, triton.cdiv(N, Br))
    _sliding_window_attn_fwd[grid](
        q, k, v, o, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        N,
        scale=scale, Br=Br, Bc=Bc, D=D, WINDOW=window,
        num_warps=4 if D <= 64 else 8,
        num_stages=3,
    )
    return o
