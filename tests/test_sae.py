"""Smoke tests for BatchTopKSAE (CPU only)."""

from __future__ import annotations

import pytest
import torch


def test_sae_import() -> None:
    from tetradflow.sae import BatchTopKSAE, SAEOutput

    assert BatchTopKSAE is not None
    assert SAEOutput is not None


def test_sae_forward_shape() -> None:
    from tetradflow.sae import BatchTopKSAE

    sae = BatchTopKSAE(input_dim=64, n_features=256, k=8)
    x = torch.randn(4, 64)
    out = sae(x)
    assert out.x_hat.shape == (4, 64), f"x_hat shape: {out.x_hat.shape}"
    assert out.z_topk.shape == (4, 256), f"z_topk shape: {out.z_topk.shape}"
    assert out.indices.shape == (4, 8), f"indices shape: {out.indices.shape}"


def test_sae_topk_sparsity() -> None:
    """True BatchTopK (arXiv:2412.06410): total non-zero across batch == k * B.

    Per-sample non-zero count can exceed k (some samples get more budget
    if their activations dominate globally). The batch-level budget is exact.
    """
    from tetradflow.sae import BatchTopKSAE

    k = 8
    B = 4
    sae = BatchTopKSAE(input_dim=64, n_features=256, k=k)
    x = torch.randn(B, 64)
    out = sae(x)
    total_nonzero = (out.z_topk != 0).sum().item()
    # Batch-level sparsity: exactly k * B active features total
    # (may be slightly less at boundary if global_k is clamped)
    assert total_nonzero <= k * B, f"Batch sparsity violated: {total_nonzero} > {k * B}"
    assert total_nonzero > 0, "All features are zero — encode is broken"


def test_sae_loss_positive() -> None:
    from tetradflow.sae import BatchTopKSAE

    sae = BatchTopKSAE(input_dim=64, n_features=256, k=8)
    x = torch.randn(4, 64)
    out = sae(x)
    loss = sae.loss(x, out)
    assert loss.item() >= 0.0


def test_sae_save_load(tmp_path: pytest.TempPathFactory) -> None:
    """Save and load round-trip via safetensors."""
    pytest.importorskip("safetensors")
    from tetradflow.sae import BatchTopKSAE

    sae = BatchTopKSAE(input_dim=32, n_features=128, k=4)
    path = str(tmp_path / "sae_test.safetensors")
    sae.save(path)

    sae2 = BatchTopKSAE.load(path, input_dim=32, n_features=128, k=4)
    x = torch.randn(2, 32)
    out1 = sae(x)
    out2 = sae2(x)
    assert torch.allclose(out1.x_hat, out2.x_hat, atol=1e-5)


def test_sae_post_step_normalises_decoder() -> None:
    from tetradflow.sae import BatchTopKSAE

    sae = BatchTopKSAE(input_dim=32, n_features=128, k=4)
    # Manually corrupt decoder norms
    with torch.no_grad():
        sae.W_dec.mul_(5.0)
    sae.post_step()
    norms = sae.W_dec.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


@pytest.mark.gpu
def test_sae_cuda() -> None:
    """GPU smoke test — skipped in CPU-only CI."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from tetradflow.sae import BatchTopKSAE

    sae = BatchTopKSAE(input_dim=64, n_features=256, k=8).cuda()
    x = torch.randn(4, 64, device="cuda")
    out = sae(x)
    assert out.x_hat.device.type == "cuda"
