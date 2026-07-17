# test_local.py — Lightweight local checks (no GPU required)
#
# Validates that all modules import cleanly and the PyTorch reference
# implementations produce correct shapes.
#
# Run: python tests/test_local.py

import sys
import numpy as np

def test_imports():
    """All flare modules import without error."""
    # These are CPU-only, so mock minimal torch if unavailable
    try:
        import torch
        HAS_TORCH = True
    except ImportError:
        print("⚠️  torch not installed — skipping import test")
        return False

    try:
        from flare.flash_attn_v2 import flash_attn_v2
        from flare.sliding_window import sliding_window_attn
        from flare.mla import mla_attention_ref, mla_attention
        from flare.kda import kda_attention_ref, kda_attention
        print("✅ All modules import cleanly")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_kda_ref_shapes():
    """KDA PyTorch reference produces correct output shape."""
    try:
        import torch
        from flare.kda import kda_attention_ref
    except ImportError:
        return False

    B, H, N, D = 1, 2, 16, 8
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)
    beta = torch.ones(B, H, N)

    o = kda_attention_ref(q, k, v, beta)
    assert o.shape == (B, H, N, D), f"Expected {(B, H, N, D)}, got {o.shape}"
    print(f"✅ KDA ref output shape: {o.shape}")
    return True


def test_mla_ref_shapes():
    """MLA PyTorch reference produces correct output shape."""
    try:
        import torch
        from flare.mla import mla_attention_ref
    except ImportError:
        return False

    B, H, N = 1, 2, 16
    d_compress = 4
    D = 8

    q = torch.randn(B, H, N, D)
    c_kv = torch.randn(B, N, d_compress)
    W_UK = torch.randn(d_compress, H, D)
    W_UV = torch.randn(d_compress, H, D)

    o = mla_attention_ref(q, c_kv, W_UK, W_UV)
    assert o.shape == (B, H, N, D), f"Expected {(B, H, N, D)}, got {o.shape}"
    print(f"✅ MLA ref output shape: {o.shape}")
    return True


def test_online_softmax_correctness():
    """Verify the online softmax math from the learning notes."""
    # This validates the core Flash Attention trick without needing a GPU.
    # Using the example from 03-flash-attention-v2.md
    scores = np.array([1.0, 3.0, 2.0, 4.0])

    # Standard softmax
    m = scores.max()
    exp_scores = np.exp(scores - m)
    ref_softmax = exp_scores / exp_scores.sum()

    # Online softmax (2 blocks of 2)
    # Block 1: [1, 3]
    m1 = max(1.0, 3.0)  # = 3
    l1 = np.exp(1 - m1) + np.exp(3 - m1)

    # Block 2: [2, 4]
    m_new = max(m1, max(2.0, 4.0))  # = 4
    l1_rescaled = l1 * np.exp(m1 - m_new)
    l2 = np.exp(2 - m_new) + np.exp(4 - m_new)
    l_total = l1_rescaled + l2

    # Reconstruct per-element (for block 1)
    p1 = np.exp(np.array([1.0, 3.0]) - m_new) / l_total
    p2 = np.exp(np.array([2.0, 4.0]) - m_new) / l_total
    online_softmax = np.concatenate([p1, p2])

    err = np.abs(online_softmax - ref_softmax).max()
    print(f"✅ Online softmax max error: {err:.2e}")
    assert err < 1e-10, f"Online softmax mismatch: {err}"
    return True


if __name__ == "__main__":
    # Ensure project root is in path
    sys.path.insert(0, '.')

    results = []
    results.append(("imports", test_imports()))
    results.append(("online_softmax", test_online_softmax_correctness()))

    # CPU tests for reference implementations
    try:
        import torch
        results.append(("kda_ref_shapes", test_kda_ref_shapes()))
        results.append(("mla_ref_shapes", test_mla_ref_shapes()))
    except ImportError:
        print("⚠️  torch not installed — skipping CPU shape tests")

    print("\n" + "=" * 40)
    all_pass = True
    for name, ok in results:
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")
        if not ok:
            all_pass = False

    sys.exit(0 if all_pass else 1)
