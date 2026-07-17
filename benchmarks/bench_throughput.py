"""
bench_throughput.py — Benchmark flare kernels across sequence lengths.

Measures wall-clock time for forward pass at seq_len = 512 → 8192.
Compares against PyTorch SDPA where applicable.

Run on Colab A100:
    python -m benchmarks.bench_throughput
"""

import torch
import time
import torch.nn.functional as F

from flare.flash_attn_v2 import flash_attn_v2


def bench_kernel(fn, warmup=3, iters=20):
    """Measure forward-pass latency in milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


def main():
    print("=" * 80)
    print("flare — Throughput Benchmark (A100)")
    print("=" * 80)

    B, H, D = 1, 8, 64
    configs = [
        (512, 64, 64),
        (1024, 64, 64),
        (2048, 64, 64),
        (4096, 64, 64),
        (8192, 64, 64),
        (512, 128, 128),
        (1024, 128, 128),
        (2048, 128, 128),
        (4096, 128, 128),
    ]

    print(f"\n{'N':>6s} {'D':>4s} {'Br':>4s} {'Bc':>4s} | "
          f"{'flare (ms)':>12s} {'SDPA (ms)':>12s} {'speedup':>8s}")
    print("-" * 60)

    for N, Br, Bc in configs:
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

        t_flare = bench_kernel(lambda: flash_attn_v2(q, k, v, causal=True, Br=Br, Bc=Bc))
        t_sdpa = bench_kernel(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))

        speedup = t_sdpa / t_flare if t_flare > 0 else 0
        print(f"{N:6d} {D:4d} {Br:4d} {Bc:4d} | "
              f"{t_flare:12.3f} {t_sdpa:12.3f} {speedup:7.2f}×")


if __name__ == "__main__":
    main()
