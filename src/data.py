"""Train/val/test splits, normalization, and the PyTorch Dataset/DataLoaders.

    python src/data.py    # prints split sizes + one batch's shapes

Splits are by race, never by lap:
    train  2018-2024, minus the 2024 val races
    val    every 4th 2024 race
    test   all of 2025
Drivers with no training laps (2025 rookies) are dropped from val/test.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT / "data" / "processed"

ALL_CHANNELS = ["Speed", "Throttle", "Brake", "RPM", "nGear"]
VAL_YEAR, TEST_YEAR = 2024, 2025
VAL_EVERY = 4  # rounds 4, 8, 12, ... of VAL_YEAR go to val
# force india was renamed racing point mid-2018 after the takeover, same car
CAR_ALIASES = {"Racing Point_2018": "Force India_2018"}


def load_years(years):
    """Concatenate laps_<year>.npz/.parquet. Returns X (n, 5, 512) and meta (row i = X[i])."""
    Xs, metas = [], []
    for year in years:
        npz, pq = PROCESSED / f"laps_{year}.npz", PROCESSED / f"laps_{year}.parquet"
        if not npz.exists():
            print(f"  (skipping {year}: {npz.name} not found)")
            continue
        Xs.append(np.load(npz)["X"])
        metas.append(pd.read_parquet(pq))
    X = np.concatenate(Xs)
    meta = pd.concat(metas, ignore_index=True)
    assert len(X) == len(meta)
    meta["car"] = meta["car"].replace(CAR_ALIASES)
    # a few laps have NaN gaps in rpm/gear (3 in 2020_R16), which would NaN the stats + loss
    finite = np.isfinite(X).all(axis=(1, 2))
    if not finite.all():
        print(f"  dropping {(~finite).sum()} laps with NaN telemetry")
        X, meta = X[finite], meta[finite].reset_index(drop=True)
    return X, meta


def split_masks(meta: pd.DataFrame):
    """Boolean train/val/test masks, decided per race_id."""
    is_val = (meta["year"] == VAL_YEAR) & (meta["round"] % VAL_EVERY == 0)
    is_test = meta["year"] == TEST_YEAR
    is_train = ~is_val & ~is_test
    # sanity check: no race in two splits
    for a, b in [(is_train, is_val), (is_train, is_test), (is_val, is_test)]:
        assert not set(meta.loc[a, "race_id"]) & set(meta.loc[b, "race_id"])
    return is_train.to_numpy(), is_val.to_numpy(), is_test.to_numpy()


def channel_stats(X_train: np.ndarray):
    """Per-channel mean/std over train laps only, each shape (C, 1)."""
    mean = np.mean(X_train, axis=(0, 2), keepdims=True) # shape (1, C, 1)
    std = np.std(X_train, axis=(0,2), keepdims=True) # shape (1, C, 1)
    return mean[0], std[0]


class LapDataset(Dataset):
    """One item = one lap: x (C, 512) float32, y driver id (int64)."""

    def __init__(self, X, y, mean, std):
        # normalize everything once up front instead of in __getitem__
        X = (X - mean) / std
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return (self.X[i], self.y[i])


def make_data(channels=ALL_CHANNELS, years=range(2018, 2026), batch_size=128):
    """Returns (loaders, info).

    loaders: "train"/"val"/"test" -> DataLoader (only train is shuffled)
    info:    driver vocab, per-split metadata, normalization stats
    e.g. make_data(channels=["Speed", "Throttle", "Brake"]) for the no rpm/gear baseline
    """
    X, meta = load_years(years)
    X = X[:, [ALL_CHANNELS.index(c) for c in channels]]
    tr, va, te = split_masks(meta)

    # driver vocab from train only, the model can't predict anyone it never saw
    drivers = sorted(meta.loc[tr, "driver"].unique())
    driver_to_id = {d: i for i, d in enumerate(drivers)}
    y_all = meta["driver"].map(driver_to_id).fillna(-1).astype(int).to_numpy()
    known = y_all >= 0

    mean, std = channel_stats(X[tr])
    loaders, metas = {}, {}
    for name, mask in [("train", tr), ("val", va & known), ("test", te & known)]:
        ds = LapDataset(X[mask], y_all[mask], mean, std)
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"))
        metas[name] = meta[mask].reset_index(drop=True)

    info = dict(drivers=drivers, driver_to_id=driver_to_id, channels=list(channels),
                mean=mean, std=std, meta=metas,
                unknown_test_drivers=sorted(meta.loc[te & ~known, "driver"].unique()))
    return loaders, info


if __name__ == "__main__":
    loaders, info = make_data()
    for name, dl in loaders.items():
        m = info["meta"][name]
        print(f"{name:<5} {len(dl.dataset):>7} laps  {m['race_id'].nunique():>3} races  "
              f"{m['driver'].nunique():>2} drivers  {m['car'].nunique():>3} cars")
    print("drivers in vocabulary:", len(info["drivers"]))
    print("2025 drivers with no training laps (dropped):", info["unknown_test_drivers"])
    xb, yb = next(iter(loaders["train"]))
    print("batch:", tuple(xb.shape), xb.dtype, tuple(yb.shape), yb.dtype)
    print("batch per-channel mean:", xb.mean(dim=(0, 2)).numpy().round(2))
    print("batch per-channel std: ", xb.std(dim=(0, 2)).numpy().round(2))
