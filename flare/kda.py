"""
kda.py — Kimi Delta Attention (Kimi Linear, 2025)

KDA is a linear attention variant with fine-grained per-channel gating.
Unlike softmax attention (O(N²)), linear attention maintains a recurrent
state of size (D, D) and processes each token in O(D²) time — O(N×D²) total.

This makes it efficient for very long sequences (>1M tokens) where the N²
cost of softmax attention becomes prohibitive.

Architecture (per layer):
  1. Q, K, V projections (same as standard attention)
  2. Delta rule update:
       S_t = S_{t-1} + β_t · V_t ⊗ K_t     (gated state update)
       O_t = Q_t @ S_{t-1} + β_t · (Q_t · K_t) · V_t
     where β_t is a per-head, per-channel gate.

  The "delta rule" name comes from the update being equivalent to
  online delta rule learning. This is a generalization of DeltaNet
  (Yang et al., 2024).

For a chunked implementation (processing Chunk tokens at a time):
  1. Compute intra-chunk causal linear attention (like a small attention matrix)
  2. Add contribution from the recurrent state (Q @ S)
  3. Update state with this chunk's K, V contributions

  CRITICAL: chunks are sequentially dependent (state from chunk i feeds chunk i+1).
  We launch one kernel per chunk from Python, passing state between launches.
  This is the same pattern used by flash-linear-attention and vLLM.

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
    State,           # (B, H, D, D) — read input state, write output state
    # strides
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_bb, stride_bh, stride_bn,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_sb, stride_sh, stride_sd0, stride_sd1,
    chunk_start,     # runtime: offset of this chunk
    N,
    scale: tl.constexpr,
    Chunk: tl.constexpr,
    D: tl.constexpr,
):
    """
    Single-chunk KDA forward. One program per (batch, head).
    Called sequentially from Python for each chunk.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_n = chunk_start + tl.arange(0, Chunk)  # (Chunk,)
    offs_d = tl.arange(0, D)                     # (D,)
    mask_n = offs_n < N

    Q_b = Q + pid_b * stride_qb + pid_h * stride_qh
    K_b = K + pid_b * stride_kb + pid_h * stride_kh
    V_b = V + pid_b * stride_vb + pid_h * stride_vh
    Beta_b = Beta + pid_b * stride_bb + pid_h * stride_bh
    O_b = O + pid_b * stride_ob + pid_h * stride_oh
    S_b = State + pid_b * stride_sb + pid_h * stride_sh

    # ── load Q, K, V, beta ──
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

    # ── load state (D, D) ──
    S = tl.load(
        S_b + offs_d[:, None] * stride_sd0 + offs_d[None, :] * stride_sd1,
    )  # (D, D)

    # ── 1. contribution from previous state: O_inter = Q @ S ──
    O_inter = tl.dot(Q_blk * scale, S)  # (Chunk, D) fp32

    # ── 2. intra-chunk causal linear attention ──
    # For token i: sum_{j<=i} beta_j * (Q_i · K_j) * V_j
    attn = tl.dot(Q_blk * scale, tl.trans(K_blk))  # (Chunk, Chunk) fp32
    attn = tl.where(
        tl.arange(0, Chunk)[:, None] >= tl.arange(0, Chunk)[None, :],
        attn, 0.0,
    )
    attn = attn * beta[None, :]  # scale by key's gate

    O_intra = tl.dot(attn.to(V_blk.dtype), V_blk)  # (Chunk, D) fp32

    # ── 3. total output ──
    O_final = (O_inter + O_intra).to(O_b.dtype.element_ty)

    tl.store(
        O_b + offs_n[:, None] * stride_on + offs_d[None, :] * stride_od,
        O_final,
        mask=mask_n[:, None],
    )

    # ── 4. update state: S += sum_t beta_t * V_t ⊗ K_t ──
    VK = tl.dot(
        tl.trans(V_blk * beta[:, None]).to(K_blk.dtype),
        K_blk,
    )  # (D, D) fp32
    S_new = S + VK.to(S.dtype)

    tl.store(
        S_b + offs_d[:, None] * stride_sd0 + offs_d[None, :] * stride_sd1,
        S_new,
    )


def kda_attention(q, k, v, beta=None, chunk_size=64):
    """
    Kimi Delta Attention (chunked forward pass).

    Chunks are processed SEQUENTIALLY — state from chunk i feeds chunk i+1.
    Each chunk launch parallelizes across (batch, head) only.

    Args:
        q:    (B, H, N, D)
        k:    (B, H, N, D)
        v:    (B, H, N, D)
        beta: (B, H, N) gating values. If None, defaults to 1.0.
        chunk_size: tokens per chunk.

    Returns:
        o: (B, H, N, D)
    """
    B, H, N, D = q.shape
    assert D in (16, 32, 64, 128), f"head_dim must be power-of-2 ≤128, got {D}"

    if beta is None:
        beta = torch.ones(B, H, N, dtype=torch.float32, device=q.device)

    o = torch.empty_like(q)
    state = torch.zeros(B, H, D, D, dtype=q.dtype, device=q.device)
    scale = 1.0 / (D ** 0.5)

    n_chunks = triton.cdiv(N, chunk_size)
    for c in range(n_chunks):
        chunk_start = c * chunk_size
        _kda_chunk_fwd[(B, H)](
            q, k, v, beta, o,
            state,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            state.stride(0), state.stride(1), state.stride(2), state.stride(3),
            chunk_start,
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
    S = torch.zeros(B, H, D, D, dtype=torch.float32, device=q.device)

    q_f32 = q.float()
    k_f32 = k.float()
    v_f32 = v.float()
    beta_f32 = beta.float() if beta.dtype != torch.float32 else beta

    for t in range(N):
        q_t = q_f32[:, :, t, :] * scale
        k_t = k_f32[:, :, t, :]
        v_t = v_f32[:, :, t, :]
        b_t = beta_f32[:, :, t]

        # output = Q_t @ S + beta_t * (Q_t · K_t) * V_t
        o_t = torch.einsum('bhd,bhde->bhe', q_t, S)
        o_t = o_t + b_t[:, :, None] * (q_t * k_t).sum(-1, keepdim=True) * v_t

        o[:, :, t, :] = o_t.to(q.dtype)

        # state update: S += beta_t * V_t ⊗ K_t
        S = S + b_t[:, :, None, None] * torch.einsum('bhd,bhe->bhde', v_t, k_t)

    return o
