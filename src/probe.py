"""Leakage probe: how much car info is in the embedding z?

    python src/probe.py naive_fcn no_rpm_gear

Per driver, a linear probe tries to tell their 2022 car from their 2023 car using
only z. The driver is fixed, so anything above 50% is car(-season) info in z.
Lower = better. Probe train/test is split by race (every 4th round is test), and
the result is the lap-weighted mean over drivers. random_init (untrained FCN) is
the reference for how much car info a random projection already shows.

Caveat: two seasons also differ in non-car stuff (style changes, tyres), so this
is an upper bound on leakage. Mostly useful for comparing models.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from data import LapDataset, make_data
from evaluate import predict
from models import FCN

PROJECT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT / "runs"
YEAR_A, YEAR_B = 2022, 2023  # same regs, both fully in train
MIN_LAPS = 300


def probe_accuracy(z_train, y_train, z_test, y_test) -> float:
    """Logistic regression on standardized z, returns test acc."""
    # scaler fit on probe-train only
    scaler = StandardScaler().fit(z_train)
    z_train_scaled = scaler.transform(z_train)
    z_test_scaled = scaler.transform(z_test)
    clf = LogisticRegression(max_iter=1000).fit(z_train_scaled, y_train)

    return clf.score(z_test_scaled, y_test)


def embeddings(model, X, mean, std, device):
    ds = LapDataset(X, np.zeros(len(X), dtype=np.int64), mean, std)
    _, z = predict(model, DataLoader(ds, batch_size=512, shuffle=False), device)
    return z


def leakage(z, meta):
    """Per-driver car probes. Returns (lap-weighted mean acc, per-driver dict)."""
    accs, weights = {}, {}
    for driver, g in meta.groupby("driver"):
        counts = g["year"].value_counts()
        if counts.get(YEAR_A, 0) < MIN_LAPS or counts.get(YEAR_B, 0) < MIN_LAPS:
            continue
        idx = g.index.to_numpy()
        y = (g["year"] == YEAR_B).to_numpy().astype(int)
        test = (g["round"] % 4 == 0).to_numpy()
        accs[driver] = probe_accuracy(z[idx[~test]], y[~test], z[idx[test]], y[test])
        weights[driver] = test.sum()
    w = np.array([weights[d] for d in accs])
    mean_acc = float(np.average(list(accs.values()), weights=w))
    return mean_acc, {d: round(a, 3) for d, a in accs.items()}


def main():
    tags = sys.argv[1:] or ["naive_fcn"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cache, results = {}, {}
    for tag in ["random_init"] + tags:
        cfg_tag = tags[0] if tag == "random_init" else tag
        config = json.loads((RUNS_DIR / cfg_tag / "config.json").read_text())
        key = tuple(config["channels"])
        if key not in cache:
            _, info = make_data(channels=config["channels"])
            X_all, meta_all = cache_years(config["channels"])
            cache[key] = (info, X_all, meta_all)
        info, X, meta = cache[key]

        torch.manual_seed(0)
        model = FCN(len(config["channels"]), len(config["drivers"])).to(device)
        if tag != "random_init":
            model.load_state_dict(torch.load(RUNS_DIR / tag / "best.pt", map_location=device))
        z = embeddings(model, X, info["mean"], info["std"], device)
        acc, per_driver = leakage(z, meta)
        results[tag] = acc
        print(f"{tag:<14} car-leakage probe accuracy {100 * acc:.1f}%  "
              f"(chance 50%, {len(per_driver)} drivers)")
        print("   per driver:", per_driver)
        if tag != "random_init":
            (RUNS_DIR / tag / "leakage.json").write_text(json.dumps(
                dict(leakage=acc, per_driver=per_driver, years=[YEAR_A, YEAR_B]), indent=2))

    print("\nLeakage (lower = less car information in z):")
    for tag, acc in results.items():
        print(f"  {tag:<14} {100 * acc:.1f}")


def cache_years(channels):
    """X + meta for YEAR_A and YEAR_B, restricted to the given channels."""
    from data import ALL_CHANNELS, load_years
    X, meta = load_years([YEAR_A, YEAR_B])
    return X[:, [ALL_CHANNELS.index(c) for c in channels]], meta


if __name__ == "__main__":
    main()
