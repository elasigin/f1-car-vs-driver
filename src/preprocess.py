"""Turn every clean race lap into a fixed-length (5, 512) array + a metadata row.

    python src/preprocess.py --years 2024 --limit 2   # quick test, 2 races
    python src/preprocess.py                          # all of 2018-2025 (~7 min)

Writes per season to data/processed/ (rows aligned):
    laps_<year>.npz      X, float32 (n_laps, 5, 512)
    laps_<year>.parquet  driver, car, race, tyres, lap time, ...

Splitting + normalization happen at load time in data.py, so the same processed
files work for any split.
"""
import argparse
import csv
import logging
import time
import warnings
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT / "data" / "cache"
OUT_DIR = PROJECT / "data" / "processed"
DOWNLOAD_LOG = PROJECT / "data" / "download_log.csv"

CHANNELS = ["Speed", "Throttle", "Brake", "RPM", "nGear"]
N_POINTS = 512
MIN_SAMPLES = 150  # a ~90s lap at 4Hz is ~350 samples, way fewer = telemetry dropout


def clean_laps(laps: fastf1.core.Laps) -> fastf1.core.Laps:
    """Keep green-flag racing laps: no in/out laps, no SC/VSC/yellows, within 107%
    of the fastest lap, accurate timing, not lap 1, no inters/wets."""
    laps = laps.pick_wo_box().pick_track_status("1").pick_quicklaps(1.07)

    keep = ((laps['IsAccurate'] == True) & (laps['LapNumber'] > 1)) & (~laps['Compound'].isin(["INTERMEDIATE", "WET"]))

    return laps[keep]


def resample_lap(tel: pd.DataFrame, n_points: int = N_POINTS) -> np.ndarray:
    """Interpolate one lap's telemetry onto n_points evenly spaced positions.
    Returns float32 of shape (len(CHANNELS), n_points)."""
    dist = tel['Distance'].to_numpy(dtype=float)
    dist = (dist - dist.min()) / (dist.max() - dist.min())  # min-max to [0, 1]
    x_grid = np.linspace(0, 1, n_points)
    rows = []
    for channel in CHANNELS:
        rows.append(np.interp(x_grid, dist, tel[channel].to_numpy(dtype=float))) # dtype=float also handles the brake bool
    return np.stack(rows, axis=0).astype(np.float32) # one row per channel


def lap_metadata(lap: pd.Series, year: int, rnd: int, event: str) -> dict:
    """Label/metadata row saved next to the lap's array. driver is the class label,
    car (team_year) is what the adversary predicts. tyres + lap time are just for analysis."""
    driver = lap["Driver"]
    team = lap["Team"]
    car = f"{team}_{year}"

    race_id = f"{year}_R{rnd:02d}"
    lap_number = lap["LapNumber"]
    compound = lap["Compound"]
    tyre_life = lap["TyreLife"]
    lap_time = lap["LapTime"].total_seconds()

    metadata = {
        "driver": driver,
        "team": team,
        "car": car,
        "year": year,
        "round": rnd,
        "event": event,
        "race_id": race_id,
        "lap_number": lap_number,
        "compound": compound,
        "tyre_life": tyre_life,
        "lap_time_s": lap_time,
    }
    return metadata


def process_race(year: int, rnd: int, event: str):
    """Load one race from the cache, return (X, meta_rows, stats)."""
    session = fastf1.get_session(year, rnd, "R")
    session.load(telemetry=True, weather=False, messages=False)

    laps = clean_laps(session.laps)
    arrays, rows = [], []
    n_short = 0
    for _, lap in laps.iterlaps():
        tel = lap.get_car_data().add_distance()
        if len(tel) < MIN_SAMPLES:  # dropout, skip instead of interpolating garbage
            n_short += 1
            continue
        arrays.append(resample_lap(tel))
        rows.append(lap_metadata(lap, year, rnd, event))

    X = np.stack(arrays) if arrays else np.empty((0, len(CHANNELS), N_POINTS), np.float32)
    stats = dict(raw=len(session.laps), clean=len(laps), short=n_short, kept=len(rows))
    return X, rows, stats


def races_to_process(years):
    """Races that downloaded OK according to download_log.csv (deduped)."""
    races = {}
    with open(DOWNLOAD_LOG) as f:
        for r in csv.DictReader(f):
            if r["status"] == "ok" and int(r["year"]) in years:
                races[(int(r["year"]), int(r["round"]))] = r["event"]
    return sorted(races.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2018, 2026)))
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N races per year (for testing)")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for year in args.years:
        races = [r for r in races_to_process({year})][: args.limit]
        Xs, meta = [], []
        t_year = time.time()
        for (yr, rnd), event in races:
            t0 = time.time()
            try:
                X, rows, st = process_race(yr, rnd, event)
            except Exception as exc:  # don't let one bad race kill the whole run
                print(f"{yr} R{rnd:02d} {event:<28} ERROR {exc}", flush=True)
                continue
            Xs.append(X)
            meta.extend(rows)
            print(f"{yr} R{rnd:02d} {event:<28} raw={st['raw']:<5} clean={st['clean']:<5} "
                  f"short={st['short']:<3} kept={st['kept']:<5} {time.time() - t0:.1f}s",
                  flush=True)

        if not meta:
            print(f"{year}: no laps kept, nothing saved")
            continue
        X_year = np.concatenate(Xs).astype(np.float32)
        meta_year = pd.DataFrame(meta)
        assert len(X_year) == len(meta_year), "arrays and metadata are misaligned"
        np.savez_compressed(OUT_DIR / f"laps_{year}.npz", X=X_year)
        meta_year.to_parquet(OUT_DIR / f"laps_{year}.parquet", index=False)

        # quick summary per season
        print(f"\n== {year}: {len(meta_year)} laps from {meta_year['race_id'].nunique()} races, "
              f"{meta_year['driver'].nunique()} drivers, {meta_year['car'].nunique()} cars, "
              f"X {X_year.shape}, {time.time() - t_year:.0f}s")
        print("   cars:", ", ".join(sorted(meta_year["car"].unique())))
        print("   laps per driver:",
              meta_year["driver"].value_counts().sort_index().to_dict(), "\n", flush=True)


if __name__ == "__main__":
    main()
