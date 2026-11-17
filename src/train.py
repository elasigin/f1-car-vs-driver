"""Train the FCN driver classifier, keep the best-val checkpoint.

    python src/train.py --tag naive_fcn
    python src/train.py --tag no_rpm_gear --channels Speed Throttle Brake
    python src/train.py --tag smoke --epochs 1 --max-batches 20    # quick test

Writes runs/<tag>/best.pt, history.json, config.json.
~40s/epoch on the M3 Pro GPU (MPS), so 30 epochs is ~20 min.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from data import ALL_CHANNELS, make_data
from models import FCN

PROJECT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT / "runs"


def train_one_epoch(model, loader, optimizer, loss_fn, device, max_batches=None):
    """One pass over the train set. Returns (mean loss, acc)."""
    model.train()
    total_loss, total_correct, total_seen = 0, 0, 0

    for batch_index, (xb, yb) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        xb = xb.to(device)
        yb = yb.to(device)
        logits, z = model(xb)
        loss = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # weight by batch size since the last batch is smaller
        total_loss += loss.item() * len(yb)
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        total_seen += len(yb)
    return (total_loss / total_seen, total_correct / total_seen)


def evaluate(model, loader, loss_fn, device):
    """Mean loss + acc on a held-out loader, no weight updates."""
    model.eval()
    total_loss, total_correct, total_seen = 0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits, z = model(xb)
            loss = loss_fn(logits, yb)

            total_loss += loss.item() * len(yb)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_seen += len(yb)

    return (total_loss / total_seen, total_correct / total_seen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="run name, e.g. naive_fcn")
    parser.add_argument("--channels", nargs="+", default=ALL_CHANNELS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="limit train batches per epoch (smoke tests only)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = RUNS_DIR / args.tag
    out.mkdir(parents=True, exist_ok=True)

    loaders, info = make_data(channels=args.channels, batch_size=args.batch_size)
    model = FCN(in_channels=len(args.channels), n_drivers=len(info["drivers"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    print(f"{args.tag}: {len(loaders['train'].dataset)} train laps, "
          f"{len(info['drivers'])} drivers, channels={args.channels}, device={device}")

    config = dict(vars(args), drivers=info["drivers"], n_params=sum(
        p.numel() for p in model.parameters()))
    (out / "config.json").write_text(json.dumps(config, indent=2))

    history, best_acc = [], -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], optimizer, loss_fn,
                                          device, args.max_batches)
        va_loss, va_acc = evaluate(model, loaders["val"], loss_fn, device)
        is_best = va_acc > best_acc
        if is_best:
            best_acc = va_acc
            torch.save(model.state_dict(), out / "best.pt")
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc))
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(f"epoch {epoch:>2}  train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} {'*' if is_best else ' '} "
              f"{time.time() - t0:.0f}s", flush=True)

    print(f"\nBest val acc {best_acc:.3f}. Saved to {out / 'best.pt'}")


if __name__ == "__main__":
    main()
