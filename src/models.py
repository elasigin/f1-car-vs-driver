"""FCN (Wang et al. 2017) for lap -> driver classification.

    python src/models.py    # shape + param count checks on random input

input (batch, C, 512) -> 3 conv blocks (128/256/128 filters, k=8/5/3)
-> global avg pool -> z (batch, 128) -> linear -> logits (batch, n_drivers)

forward() returns (logits, z) since the leakage probe and the adversary need z.
"""
import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int, kernel_size: int) -> nn.Sequential:
    """Conv1d -> BatchNorm1d -> ReLU, length stays the same."""
    conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding="same")
    return nn.Sequential(conv, nn.BatchNorm1d(out_ch), nn.ReLU())


class FCN(nn.Module):
    """Lap telemetry -> (driver logits, embedding z)."""

    def __init__(self, in_channels: int, n_drivers: int, emb_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_channels, 128, 8),
            conv_block(128, 256, 5),
            conv_block(256, emb_dim, 3),
            nn.AdaptiveAvgPool1d(1),  # avg over the lap -> (batch, emb_dim, 1)
            nn.Flatten()
        )
        self.head = nn.Linear(emb_dim, n_drivers)


    def forward(self, x: torch.Tensor):
        z = self.features(x)
        logits = self.head(z)
        return logits, z


if __name__ == "__main__":
    torch.manual_seed(0)
    for in_ch in (5, 3):
        model = FCN(in_channels=in_ch, n_drivers=39)
        x = torch.randn(128, in_ch, 512)
        logits, z = model(x)
        assert logits.shape == (128, 39), logits.shape
        assert z.shape == (128, 128), z.shape
        n_params = sum(p.numel() for p in model.parameters())
        print(f"in_channels={in_ch}: logits {tuple(logits.shape)}, z {tuple(z.shape)}, "
              f"{n_params:,} parameters")
        # GAP means any lap length works
        assert model(torch.randn(2, in_ch, 300))[0].shape == (2, 39)

    if torch.backends.mps.is_available():
        model = FCN(5, 39).to("mps")
        logits, z = model(torch.randn(128, 5, 512, device="mps"))
        print("MPS forward pass OK:", logits.device)
    print("All model checks passed.")
