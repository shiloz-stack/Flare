<div align="center">

# flare

**From-scratch Triton kernels for modern attention variants — benchmarked head-to-head on A100.**

FlashAttention v2 · Sliding-Window · MLA (DeepSeek) · KDA (Kimi Linear)

</div>

---

## Why

Every modern LLM uses a custom attention kernel. The choices are multiplying:
standard softmax (FlashAttention), compressed KV cache (MLA), linear/recurrent
(KDA/Mamba2). Most engineers treat these as black boxes. **flare implements all
of them from scratch in Triton** to understand the tradeoffs at the kernel level.

No wrapper around `flash-attn`. No copy-paste from tutorials. Each kernel is
written from the algorithm — tiling, online softmax, recurrent state — and
verified against PyTorch reference implementations.

## What's Inside

| Kernel | Type | Paper | Complexity | Key Trick |
|--------|------|-------|------------|-----------|
| **FlashAttention v2** | Softmax | [Dao 2023](https://arxiv.org/abs/2307.08691) | O(N²d) | Online softmax + tiling — never materialize N×N matrix |
| **Sliding-Window** | Softmax | [Mistral 2023](https://arxiv.org/abs/2310.06825) | O(N×W) | FA v2 with local-window mask instead of causal |
| **MLA** | Softmax | [DeepSeek-V2 2024](https://arxiv.org/abs/2405.04456) | O(N²d_r) | Compress KV cache via low-rank latent, up-project inside kernel |
| **KDA** | Linear | [Kimi Linear 2025](https://github.com/fla-org/flash-linear-attention) | O(N×d²) | Chunked recurrent state with per-channel gating (delta rule) |

### The Core Algorithms

**FlashAttention v2** — The bottleneck in standard attention isn't FLOPs,
it's memory bandwidth: the N×N attention matrix doesn't fit in SRAM, so the GPU
shuttles it to HBM 4 times. Flash Attention tiles the computation into blocks
that fit in SRAM and uses **online softmax** (incremental running max + rescale)
to compute exact softmax without ever materializing the full matrix.

**MLA** — In multi-head attention, the KV cache grows as `2 × H × D × N` per
layer. DeepSeek-V2 compresses this by projecting K and V through a low-rank
bottleneck (`d_compress ≪ H × D`), caching only the compressed latent. The
up-projection happens **inside the attention kernel**, so full K and V are never
written to HBM. For a 128K context with H=128, D=128, d_compress=512: **98.4%
KV cache reduction**.

**KDA** — Linear attention replaces softmax with a recurrent state update,
dropping complexity from O(N²) to O(N) at fixed D. Kimi Delta Attention adds
fine-grained per-channel gating via the delta rule, generalizing DeltaNet.
The chunked implementation processes B tokens at a time, computing intra-chunk
attention while carrying state across chunks.

## Quickstart

```python
from flare.flash_attn_v2 import flash_attn_v2

# q, k, v: (B, H, N, D) fp16 tensors on GPU
o = flash_attn_v2(q, k, v, causal=True)
```

See [`notebooks/flare_benchmark.ipynb`](notebooks/flare_benchmark.ipynb) for a
one-click Colab notebook that runs all correctness tests and benchmarks on A100.

## Correctness

All kernels are verified against PyTorch reference implementations:

| Kernel | Reference | Tolerance |
|--------|-----------|-----------|
| FlashAttention v2 | `torch.nn.functional.scaled_dot_product_attention` | 1e-2 |
| Sliding-Window | Manual loop with window mask | 1e-2 |
| MLA | PyTorch eager-mode up-projection + SDPA | 1e-2 |
| KDA | Sequential per-token recurrence | 2e-2 |

> Tolerance is 1e-2 (not 1e-6) because attention involves `exp()` and softmax
> normalization, which amplify fp16 rounding errors. PyTorch's own SDPA has the
> same tolerance when compared across backends.

## Project Structure

```
flare/
├── flare/
│   ├── flash_attn_v2.py    # Online softmax + tiling (170 lines)
│   ├── sliding_window.py   # FA v2 + local window mask (141 lines)
│   ├── mla.py              # Low-rank KV compression (210 lines)
│   └── kda.py              # Chunked recurrent delta rule (230 lines)
├── benchmarks/
│   ├── bench_correctness.py
│   └── bench_throughput.py
├── notebooks/
│   └── flare_benchmark.ipynb   # One-click Colab (A100)
└── tests/
    └── test_local.py           # CPU-only math verification
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- Triton 2.0+
- NVIDIA GPU (Ampere+ recommended for bf16/fp16 tensor cores)

## License

MIT
