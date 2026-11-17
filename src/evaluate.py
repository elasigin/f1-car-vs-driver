"""Evaluate trained checkpoints on val (2024) and test (2025).

    python src/evaluate.py naive_fcn no_rpm_gear    # several runs -> one table

Writes runs/<tag>/results.json and predictions.parquet (one row per val/test lap).

Metrics:
    val_acc         39-way acc on the 2024 val races (same car-seasons as train)
    val_teammate    only choose between the drivers of that lap's car (~50% chance)
    tierA_acc       2025, drivers who stayed at their team (new car-season, mild shift)
    tierB_acc       2025, drivers who switched teams (HAM, SAI, OCO, HUL, TSU), the main one
    tierB_teammate  switchers, choosing only among their new car's drivers (HAM or LEC?)
    predicted_as    who each switcher gets predicted as, e.g. HAM -> LEC means it learned the car
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import make_data
from models import FCN

PROJECT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT / "runs"
# tiers only count drivers with enough training laps. near-rookies (BEA 88, COL 188,
# LAW 321) can't be recognized in any car, that's missing data not the car shortcut
MIN_TRAIN_LAPS = 1000

# map renamed teams to one constructor so e.g. RB -> Racing Bulls isn't a "switch"
TEAM_FAMILY = {
    "Toro Rosso": "Faenza", "AlphaTauri": "Faenza", "RB": "Faenza", "Racing Bulls": "Faenza",
    "Force India": "Silverstone", "Racing Point": "Silverstone", "Aston Martin": "Silverstone",
    "Sauber": "Hinwil", "Alfa Romeo Racing": "Hinwil", "Alfa Romeo": "Hinwil",
    "Kick Sauber": "Hinwil",
    "Renault": "Enstone", "Alpine": "Enstone",
}


def family(team: str) -> str:
    return TEAM_FAMILY.get(team, team)


def teammate_correct(logits: np.ndarray, meta: pd.DataFrame, driver_to_id: dict) -> np.ndarray:
    """Per-lap correctness when only the drivers of that lap's car are allowed.
    Returns a bool array (n_laps,)."""
    candidate_drivers = meta.groupby("car")["driver"].unique()
    n_laps = meta.shape[0]
    allowed = np.zeros(logits.shape, dtype=bool)
    for i in range(n_laps):
        car = meta["car"].iloc[i] # which car lap i was driven in
        for candidate in candidate_drivers[car]: # that car's drivers only
            allowed[i, driver_to_id[candidate]] = True

    # -inf can never win the argmax
    masked = np.where(allowed, logits, -np.inf)
    preds = masked.argmax(axis=1)
    true_ids = meta["driver"].map(driver_to_id).to_numpy()

    return preds == true_ids


def predicted_as(preds: pd.DataFrame, driver: str, top: int = 3) -> dict:
    """Top predicted drivers (as fractions) for one true driver's laps."""
    keep = preds["driver"] == driver
    top_entries = preds.loc[keep, "pred"].value_counts(normalize=True).head(top)

    return top_entries.round(3).to_dict()


@torch.no_grad()
def predict(model, loader, device):
    """Logits + embeddings z for every lap, in loader order."""
    model.eval()
    all_logits, all_z = [], []
    for xb, _ in loader:
        logits, z = model(xb.to(device))
        all_logits.append(logits.cpu())
        all_z.append(z.cpu())
    return torch.cat(all_logits).numpy(), torch.cat(all_z).numpy()


def switch_mask(train_meta: pd.DataFrame, test_meta: pd.DataFrame) -> np.ndarray:
    """True for test laps where the driver's team differs from their last team in train."""
    last = (train_meta.sort_values(["year", "round"])
            .groupby("driver")["team"].last().map(family))
    return (test_meta["team"].map(family) != test_meta["driver"].map(last)).to_numpy()


def evaluate_run(tag: str, cache: dict):
    out = RUNS_DIR / tag
    config = json.loads((out / "config.json").read_text())
    key = tuple(config["channels"])
    if key not in cache:  # data only depends on the channels, load once per channel set
        cache[key] = make_data(channels=config["channels"], batch_size=512)
    loaders, info = cache[key]
    assert info["drivers"] == config["drivers"], "driver vocabulary changed since training"
    d2i, drivers = info["driver_to_id"], np.array(info["drivers"])

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = FCN(len(config["channels"]), len(drivers)).to(device)
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))

    res, frames = {"tag": tag, "channels": config["channels"]}, []
    for split in ["val", "test"]:
        logits, _ = predict(model, loaders[split], device)
        meta = info["meta"][split].copy()
        meta["pred"] = drivers[logits.argmax(axis=1)]
        meta["correct"] = meta["pred"] == meta["driver"]
        meta["split"] = split
        tm = teammate_correct(logits, meta, d2i)
        # cars w/ a rookie teammate (e.g. merc 2025: RUS + ANT) only have 1 candidate,
        # which is trivially right, so only score teammate acc on cars w/ 2+ candidates
        multi = (meta.groupby("car")["driver"].transform("nunique") >= 2).to_numpy()
        if split == "val":
            res["val_acc"] = meta["correct"].mean()
            res["val_teammate"] = tm[multi].mean()
        else:
            sw = switch_mask(info["meta"]["train"], meta)
            meta["tier"] = np.where(sw, "B", "A")
            n_train = info["meta"]["train"]["driver"].value_counts()
            est = (meta["driver"].map(n_train).fillna(0) >= MIN_TRAIN_LAPS).to_numpy()
            A, B = ~sw & est, sw & est
            meta["established"] = est
            res["tierA_acc"] = meta.loc[A, "correct"].mean()
            res["tierA_teammate"] = tm[A & multi].mean()
            res["tierB_acc"] = meta.loc[B, "correct"].mean()
            res["tierB_teammate"] = tm[B & multi].mean()
            # same tiers incl. near-rookies, just for reference
            res["tierA_acc_all"] = meta.loc[~sw, "correct"].mean()
            res["tierB_acc_all"] = meta.loc[sw, "correct"].mean()
            res["switchers"] = sorted(meta.loc[B, "driver"].unique())
            res["near_rookies"] = {d: int(n_train.get(d, 0)) for d in
                                   sorted(meta.loc[~est, "driver"].unique())}
            all_sw = sorted(meta.loc[sw, "driver"].unique())
            res["predicted_as"] = {d: predicted_as(meta[sw], d) for d in all_sw}
            res["per_switcher_acc"] = meta[sw].groupby("driver")["correct"].mean().round(3).to_dict()
            res["n_laps"] = {"val": int(len(info["meta"]["val"])), "tierA": int(A.sum()),
                             "tierB": int(B.sum()), "near_rookie": int((~est).sum())}
        frames.append(meta)

    res = {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
           for k, v in res.items()}
    (out / "results.json").write_text(json.dumps(res, indent=2))
    pd.concat(frames, ignore_index=True).to_parquet(out / "predictions.parquet", index=False)
    return res


def main():
    tags = sys.argv[1:] or ["naive_fcn"]
    cache, rows = {}, []
    for tag in tags:
        res = evaluate_run(tag, cache)
        rows.append(res)
        print(f"\n=== {tag}  ({len(res['channels'])} channels)  laps: {res['n_laps']}")
        print(f"Tier B (established switchers): {res['switchers']}")
        print(f"near-rookies (< {MIN_TRAIN_LAPS} training laps, excluded from tiers):",
              res["near_rookies"])
        print(f"tiers incl. near-rookies: A {100 * res['tierA_acc_all']:.1f}, "
              f"B {100 * res['tierB_acc_all']:.1f}")
        print("per-switcher accuracy:", res["per_switcher_acc"])
        for d, top in res["predicted_as"].items():
            print(f"  {d} predicted as: {top}")

    cols = ["val_acc", "val_teammate", "tierA_acc", "tierA_teammate", "tierB_acc",
            "tierB_teammate"]
    table = pd.DataFrame(rows).set_index("tag")[cols]
    print("\n", (table * 100).round(1).to_string())
    print("\nLaTeX rows:")
    for tag, r in table.iterrows():
        print(tag.replace("_", r"\_") + " & " + " & ".join(f"{100 * v:.1f}" for v in r) + r" \\")


if __name__ == "__main__":
    main()
