# test_kda_chunk_correctness.py
#
# Standalone test: verify KDA Triton chunked kernel matches sequential reference.
# This one is special because the chunked kernel has a correctness issue
# (intra-chunk causal attention needs careful state handling).
#
# Run on Colab: python tests/test_kda_chunk_correctness.py

import torch
import sys
sys.path.insert(0, '.')

from flare.kda import kda_attention, kda_attention_ref


def main():
    torch.manual_seed(42)
    B, H, N, D = 1, 2, 64, 32
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    beta = torch.rand(B, H, N, device='cuda', dtype=torch.float16) * 0.5 + 0.5

    print(f"Config: B={B} H={H} N={N} D={D}")
    print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}, beta: {beta.shape}")

    # Sequential reference (exact)
    ref = kda_attention_ref(q, k, v, beta)
    print(f"\nref output shape: {ref.shape}")
    print(f"ref[0,0,0,:5]: {ref[0,0,0,:5]}")

    # Chunked Triton kernel
    for chunk in [16, 32, 64]:
        try:
            tri = kda_attention(q, k, v, beta, chunk_size=chunk)
            err = (tri - ref).abs().max().item()
            print(f"\nchunk={chunk}: max error = {err:.6e}  {'✅ PASS' if err < 2e-2 else '❌ FAIL'}")
            if err >= 2e-2:
                diff = (tri - ref).abs()
                print(f"  error breakdown: mean={diff.mean().item():.4e}, max={diff.max().item():.4e}")
                # show where the error is
                max_idx = diff.argmax().item()
                print(f"  max error at flat index {max_idx}")
        except Exception as e:
            print(f"\nchunk={chunk}: ❌ ERROR: {e}")


if __name__ == "__main__":
    main()
