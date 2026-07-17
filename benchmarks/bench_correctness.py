"""
bench_correctness.py — Verify flare kernels against PyTorch reference implementations.

Run on Colab A100:
    python -m benchmarks.bench_correctness

Each test checks:
  1. Triton kernel output vs PyTorch reference (max absolute error)
  2. Tolerance: 1e-2 for fp16 (attn involves exp + softmax, so slightly looser)
"""

import torch
import torch.nn.functional as F

from flare.flash_attn_v2 import flash_attn_v2
from flare.sliding_window import sliding_window_attn
from flare.mla import mla_attention, mla_attention_ref
from flare.kda import kda_attention, kda_attention_ref


def test_flash_attn_v2():
    """FlashAttention v2 vs PyTorch SDPA."""
    print("=" * 60)
    print("FlashAttention v2 — correctness")
    print("=" * 60)

    torch.manual_seed(42)
    B, H, N, D = 2, 4, 256, 64
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    # PyTorch reference (SDPA = Scaled Dot-Product Attention)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # Triton kernel (non-causal first)
    tri_non_causal = flash_attn_v2(q, k, v, causal=False)
    ref_non_causal = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    err_nc = (tri_non_causal - ref_non_causal).abs().max().item()

    # Triton kernel (causal)
    tri_causal = flash_attn_v2(q, k, v, causal=True)
    err_c = (tri_causal - ref).abs().max().item()

    print(f"  Config: B={B} H={H} N={N} D={D} dtype=fp16")
    print(f"  Non-causal max error: {err_nc:.6e}  {'✅ PASS' if err_nc < 1e-2 else '❌ FAIL'}")
    print(f"  Causal     max error: {err_c:.6e}  {'✅ PASS' if err_c < 1e-2 else '❌ FAIL'}")
    return err_nc < 1e-2 and err_c < 1e-2


def test_sliding_window():
    """Sliding-window attention vs manual PyTorch."""
    print("\n" + "=" * 60)
    print("Sliding-Window Attention — correctness")
    print("=" * 60)

    torch.manual_seed(42)
    B, H, N, D = 2, 4, 256, 64
    W = 32  # window size
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    # Manual reference
    scale = 1.0 / (D ** 0.5)
    ref = torch.zeros_like(q)
    for b in range(B):
        for h in range(H):
            for i in range(N):
                j_start = max(0, i - W + 1)
                s = (q[b, h, i] @ k[b, h, j_start:i+1].T) * scale
                p = torch.softmax(s, dim=-1)
                ref[b, h, i] = p @ v[b, h, j_start:i+1]

    tri = sliding_window_attn(q, k, v, window=W)
    err = (tri - ref).abs().max().item()

    print(f"  Config: B={B} H={H} N={N} D={D} W={W} dtype=fp16")
    print(f"  Max error: {err:.6e}  {'✅ PASS' if err < 1e-2 else '❌ FAIL'}")
    return err < 1e-2


def test_mla():
    """MLA vs PyTorch reference."""
    print("\n" + "=" * 60)
    print("Multi-head Latent Attention — correctness")
    print("=" * 60)

    torch.manual_seed(42)
    B, H, N = 2, 4, 256
    d_compress = 32  # compressed dimension (much smaller than H * D)
    D = 64

    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    c_kv = torch.randn(B, N, d_compress, device='cuda', dtype=torch.float16)
    W_UK = torch.randn(d_compress, H, D, device='cuda', dtype=torch.float16)
    W_UV = torch.randn(d_compress, H, D, device='cuda', dtype=torch.float16)

    ref = mla_attention_ref(q, c_kv, W_UK, W_UV)
    tri = mla_attention(q, c_kv, W_UK, W_UV)
    err = (tri - ref).abs().max().item()

    kv_cache_std = 2 * H * D * N * 2  # bytes (fp16)
    kv_cache_mla = d_compress * N * 2
    print(f"  Config: B={B} H={H} N={N} D={D} d_compress={d_compress} dtype=fp16")
    print(f"  KV cache: standard={kv_cache_std/1024:.0f}KB  MLA={kv_cache_mla/1024:.0f}KB  "
          f"({kv_cache_mla/kv_cache_std*100:.1f}%)")
    print(f"  Max error: {err:.6e}  {'✅ PASS' if err < 1e-2 else '❌ FAIL'}")
    return err < 1e-2


def test_kda():
    """KDA vs sequential PyTorch reference."""
    print("\n" + "=" * 60)
    print("Kimi Delta Attention — correctness")
    print("=" * 60)

    torch.manual_seed(42)
    B, H, N, D = 1, 2, 128, 32
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    beta = torch.rand(B, H, N, device='cuda', dtype=torch.float16) * 0.5 + 0.5  # [0.5, 1.0]

    ref = kda_attention_ref(q, k, v, beta)
    tri = kda_attention(q, k, v, beta, chunk_size=64)
    err = (tri - ref).abs().max().item()

    print(f"  Config: B={B} H={H} N={N} D={D} dtype=fp16")
    print(f"  Max error: {err:.6e}  {'✅ PASS' if err < 2e-2 else '❌ FAIL'}")
    return err < 2e-2


def test_flash_attn_edge_cases():
    """Edge cases: seq_len not divisible by block size."""
    print("\n" + "=" * 60)
    print("FlashAttention v2 — edge cases")
    print("=" * 60)

    all_pass = True
    for N in [63, 65, 127, 128, 129, 200]:
        q = torch.randn(1, 2, N, 32, device='cuda', dtype=torch.float16)
        k = torch.randn(1, 2, N, 32, device='cuda', dtype=torch.float16)
        v = torch.randn(1, 2, N, 32, device='cuda', dtype=torch.float16)

        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        tri = flash_attn_v2(q, k, v, causal=True, Br=64, Bc=64)
        err = (tri - ref).abs().max().item()
        ok = err < 1e-2
        print(f"  N={N:4d}: max error={err:.6e}  {'✅' if ok else '❌'}")
        all_pass = all_pass and ok

    return all_pass


if __name__ == "__main__":
    results = {}
    results["flash_attn_v2"] = test_flash_attn_v2()
    results["sliding_window"] = test_sliding_window()
    results["mla"] = test_mla()
    results["kda"] = test_kda()
    results["edge_cases"] = test_flash_attn_edge_cases()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {name:20s} {'✅ PASS' if passed else '❌ FAIL'}")
