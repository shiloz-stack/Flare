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
  sliding: q_idx >= k_idx AND q_idx - k_idx < WINDOW
"""

import torch
import triton
import triton.language as tl


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
    offs_d = tl.arange(0, D)

    Q_blk = tl.load(
        Q_b + q_idx[:, None] * stride_qn + offs_d[None, :] * stride_qd,
        mask=q_idx[:, None] < N, other=0.0,
    )

    m_i = tl.full([Br], -1e4, dtype=tl.float32)
    l_i = tl.full([Br], 0.0, dtype=tl.float32)
    O_acc = tl.zeros([Br, D], dtype=tl.float32)

    # Iterate over ALL K/V blocks — skip invalid ones via mask.
    # This avoids Triton JIT issues with data-dependent loop bounds.
    n_kv_blocks = tl.cdiv(N, Bc)
    for j in range(0, n_kv_blocks):
        kj = j * Bc + tl.arange(0, Bc)

        K_blk = tl.load(
            K_b + kj[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=kj[:, None] < N, other=0.0,
        )
        V_blk = tl.load(
            V_b + kj[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=kj[:, None] < N, other=0.0,
        )

        S = tl.dot(Q_blk, tl.trans(K_blk)) * scale

        # ── sliding window + causal mask ──
        mask = (q_idx[:, None] >= kj[None, :]) & \
               (q_idx[:, None] - kj[None, :] < WINDOW) & \
               (kj[None, :] < N)
        S = tl.where(mask, S, -1e4)

        # ── online softmax ──
        m_block = tl.max(S, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        # zero out masked entries in p so they don't contribute to O_acc
        p = tl.where(mask, tl.exp(S - m_new[:, None]), 0.0)

        l_i = l_i * alpha
        O_acc = O_acc * alpha[:, None]
        l_i = l_i + tl.sum(p, axis=1)
        O_acc = O_acc + tl.dot(p.to(V_blk.dtype), V_blk)
        m_i = m_new

    # ── guard against division by zero ──
    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    O_acc = O_acc / l_i_safe[:, None]

    tl.store(
        O_b + q_idx[:, None] * stride_on + offs_d[None, :] * stride_od,
        O_acc,
        mask=q_idx[:, None] < N,
    )
    lse = tl.log(l_i_safe)
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
