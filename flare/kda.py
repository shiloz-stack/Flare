"""
kda.py — Kimi Delta Attention (Kimi Linear, 2025)

KDA is a linear attention variant with fine-grained per-channel gating.
Unlike softmax attention (O(N²)), linear attention maintains a recurrent
state of size (D, D) and processes each token in O(D²) time — O(N×D²) total.

This makes it efficient for very long sequences (>1M tokens) where the N²
cost of softmax attention becomes prohibitive.

Architecture (per layer):
  1. Q, K, V projections (same as standard attention)
  2. Optional RoPE on Q, K
  3. Delta rule update:
       S_t = β_t * S_{t-1} + V_t ⊗ K_t           (gated state update)
       O_t = Q_t @ S_t                             (output)
     where β_t = σ(q_t · w_β) is a per-head, per-channel gate.

  The "delta rule" name comes from the update being equivalent to
  online delta rule learning: S_t = (I - β_t K_t K_t^T) S_{t-1} + β_t V_t K_t^T.
  This is a generalization of DeltaNet (Yang et al., 2024).

For a chunked implementation (processing B tokens at a time instead of 1):
  1. Compute K, V for the chunk
  2. Compute inter-chunk state transitions
 3. Compute intra-chunk attention with the state carried in

This Triton kernel implements the chunked forward pass.

References:
  - Kimi Linear technical report (2025)
  - Yang et al., "Gated DeltaNet: Improving Mamba2 with Delta Rule", 2024
  - https://github.com/fla-org/flash-linear-attention
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _kda_chunk_fwd(
    Q, K, V, Beta, O,
    # state carried across chunks (intra-layer, not across layers)
    State_init,
    # strides
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_bb, stride_bh, stride_bn,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_sb, stride_sh, stride_sd0, stride_sd1,
    N,
    scale: tl.constexpr,
    Chunk: tl.constexpr,
    D: tl.constexpr,
):
    """
    Chunked KDA forward. One program per (batch, head, chunk).

    Within each chunk of `Chunk` tokens:
      1. Load Q, K, V, Beta for the chunk
      2. Compute intra-chunk attention (causal, linear)
      3. Carry state forward to next chunk

    State is a (D, D) matrix per (batch, head).
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_c = tl.program_id(2)

    chunk_start = pid_c * Chunk

    # ── load Q, K, V for this chunk ──
    offs_n = chunk_start + tl.arange(0, Chunk)  # (Chunk,)
    offs_d = tl.arange(0, D)                     # (D,)
    mask_n = offs_n < N

    Q_b = Q + pid_b * stride_qb + pid_h * stride_qh
    K_b = K + pid_b * stride_kb + pid_h * stride_kh
    V_b = V + pid_b * stride_vb + pid_h * stride_vh
    Beta_b = Beta + pid_b * stride_bb + pid_h * stride_bh
    O_b = O + pid_b * stride_ob + pid_h * stride_oh

    Q_blk = tl.load(
        Q_b + offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd,
        mask=mask_n[:, None], other=0.0,
    )  # (Chunk, D)

    K_blk = tl.load(
        K_b + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        mask=mask_n[:, None], other=0.0,
    )  # (Chunk, D)

    V_blk = tl.load(
        V_b + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
        mask=mask_n[:, None], other=0.0,
    )  # (Chunk, D)

    beta = tl.load(
        Beta_b + offs_n * stride_bn,
        mask=mask_n, other=0.0,
    )  # (Chunk,)

    # ── load incoming state from previous chunk ──
    S_b = State_init + pid_b * stride_sb + pid_h * stride_sh
    # S is (D, D) — load entire matrix
    S = tl.load(
        S_b + offs_d[:, None] * stride_sd0 + offs_d[None, :] * stride_sd1,
    )  # (D, D)

    # ── compute output for this chunk ──
    # O = Q @ S (contribution from previous state)
    O_intra = tl.dot(Q_blk * scale, S)  # (Chunk, D) — fp32 accumulator

    # ── intra-chunk: causal linear attention within the chunk ──
    # For token i, attend to tokens j <= i within the chunk:
    #   O_i += sum_{j<=i} beta_j * V_j (Q_i · K_j)
    # This is a causal linear attention pattern.
    # We compute it as a (Chunk, Chunk) matrix that fits in SRAM.
    attn = tl.dot(Q_blk * scale, tl.trans(K_blk))  # (Chunk, Chunk) — fp32
    attn = tl.where(
        tl.arange(0, Chunk)[:, None] >= tl.arange(0, Chunk)[None, :],
        attn, 0.0,
    )  # causal mask
    # apply beta scaling
    attn = attn * beta[None, :]  # (Chunk, Chunk)

    O_chunk = tl.dot(attn.to(V_blk.dtype), V_blk)  # (Chunk, D) — cast attn to fp16 for dot

    # total output = contribution from state + intra-chunk
    O_final = (O_intra + O_chunk).to(O_b.dtype.element_ty)

    # ── store output ──
    tl.store(
        O_b + offs_n[:, None] * stride_on + offs_d[None, :] * stride_od,
        O_final,
        mask=mask_n[:, None],
    )

    # ── update state for next chunk ──
    # S_new = S + sum_t beta_t * V_t ⊗ K_t
    #       = S + V^T @ diag(beta) @ K
    # V_blk: (Chunk, D), K_blk: (Chunk, D)
    VK = tl.dot(
        tl.trans(V_blk * beta[:, None]).to(K_blk.dtype),
        K_blk,
    )  # (D, D) — fp32, but S is fp16 → cast
    S_new = S + VK.to(S.dtype)

    tl.store(
        S_b + offs_d[:, None] * stride_sd0 + offs_d[None, :] * stride_sd1,
        S_new,
    )


def kda_attention(q, k, v, beta=None, chunk_size=64):
    """
    Kimi Delta Attention (chunked forward pass).

    Args:
        q:    (B, H, N, D) queries
        k:    (B, H, N, D) keys
        v:    (B, H, N, D) values
        beta: (B, H, N) gating values in [0, 1]. If None, defaults to 1.0.
        chunk_size: number of tokens per chunk (larger = more parallelism).

    Returns:
        o: (B, H, N, D) outputs
    """
    B, H, N, D = q.shape
    assert D in (16, 32, 64, 128), f"head_dim must be power-of-2 ≤128, got {D}"
    assert q.shape == k.shape == v.shape

    if beta is None:
        beta = torch.ones(B, H, N, dtype=torch.float32, device=q.device)

    o = torch.empty_like(q)
    state = torch.zeros(B, H, D, D, dtype=q.dtype, device=q.device)
    scale = 1.0 / (D ** 0.5)

    grid = (B, H, triton.cdiv(N, chunk_size))
    _kda_chunk_fwd[grid](
        q, k, v, beta, o,
        state,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        N,
        scale=scale, Chunk=chunk_size, D=D,
        num_warps=4 if D <= 64 else 8,
        num_stages=2,
    )
    return o


# ─── PyTorch reference for testing ───────────────────────────────────────────

def kda_attention_ref(q, k, v, beta=None):
    """
    PyTorch reference: sequential (non-chunked) KDA forward.
    O(N×D²), but mathematically exact for verification.
    """
    B, H, N, D = q.shape
    if beta is None:
        beta = torch.ones(B, H, N, dtype=q.dtype, device=q.device)

    scale = 1.0 / (D ** 0.5)
    o = torch.empty_like(q)
    S = torch.zeros(B, H, D, D, dtype=q.dtype, device=q.device)

    for t in range(N):
        q_t = q[:, :, t, :] * scale         # (B, H, D)
        k_t = k[:, :, t, :]                   # (B, H, D)
        v_t = v[:, :, t, :]                   # (B, H, D)
        b_t = beta[:, :, t]                    # (B, H)

        # output = Q_t @ S + beta_t * (Q_t · K_t) * V_t
        o_t = torch.einsum('bhd,bhde->bhe', q_t, S)
        o_t = o_t + b_t[:, :, None] * (q_t * k_t).sum(-1, keepdim=True) * v_t

        o[:, :, t, :] = o_t

        # state update: S += beta_t * V_t ⊗ K_t
        S = S + b_t[:, :, None, None] * torch.einsum('bhd,bhe->bhde', v_t, k_t)

    return o
