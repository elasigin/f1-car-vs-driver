"""Download every F1 race (with telemetry) for a range of seasons into the FastF1 cache.

    python src/download.py --start 2018 --end 2025

Needs an F1 TV login first: python -m fastf1 auth f1tv --authenticate

FastF1 caps you at 500 API calls/hour (~38 races), so this waits out the limit
instead of crashing. A full 2018-2025 run takes ~5 hrs. Safe to re-run, cached
races load from disk. Each race's result gets appended to data/download_log.csv.
"""
import argparse
import csv
import logging
import time
import warnings
from pathlib import Path

import fastf1
from fastf1.exceptions import RateLimitExceededError

PROJECT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT / "data" / "cache"
LOG_PATH = PROJECT / "data" / "download_log.csv"
FIELDS = ["year", "round", "event", "status", "laps", "drivers", "telemetry", "seconds"]
WAIT_SECONDS = 300  # pause after hitting the hourly limit


def with_rate_limit_retry(fn, label):
    """Call fn(), waiting and retrying whenever the hourly limit is hit."""
    while True:
        try:
            return fn()
        except RateLimitExceededError:
            print(f"  rate limit reached at {label}; waiting {WAIT_SECONDS // 60} min "
                  f"({time.strftime('%H:%M')})", flush=True)
            time.sleep(WAIT_SECONDS)


def load_race(year, rnd):
    session = fastf1.get_session(year, rnd, "R")
    session.load(telemetry=True, weather=True, messages=False)
    return session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2018)
    parser.add_argument("--end", type=int, default=2025)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    new_log = not LOG_PATH.exists()
    log_file = open(LOG_PATH, "a", newline="")
    writer = csv.DictWriter(log_file, fieldnames=FIELDS)
    if new_log:
        writer.writeheader()

    n_ok = n_total = 0
    for year in range(args.end, args.start - 1, -1):  # newest season first
        schedule = with_rate_limit_retry(
            lambda: fastf1.get_event_schedule(year, include_testing=False),
            f"{year} schedule")
        for _, event in schedule.iterrows():
            rnd, name = int(event["RoundNumber"]), event["EventName"]
            t0 = time.time()
            try:
                session = with_rate_limit_retry(lambda: load_race(year, rnd),
                                                f"{year} R{rnd:02d}")
                n_laps = len(session.laps)
                n_drivers = session.laps["Driver"].nunique()
                has_tel = len(session.car_data) > 0
                status = "ok" if (n_laps > 0 and has_tel) else "missing_data"
            except Exception as exc:  # don't let one bad race kill the whole run
                n_laps, n_drivers, has_tel = 0, 0, False
                status = f"error: {exc}".replace("\n", " ")[:120]
            secs = round(time.time() - t0, 1)
            print(f"{year} R{rnd:02d} {name:<30} {status:<14} laps={n_laps:<5} "
                  f"drivers={n_drivers:<3} {secs}s", flush=True)
            writer.writerow(dict(year=year, round=rnd, event=name, status=status,
                                 laps=n_laps, drivers=n_drivers, telemetry=has_tel,
                                 seconds=secs))
            log_file.flush()
            n_total += 1
            n_ok += status == "ok"

    log_file.close()
    print(f"\nDone: {n_ok}/{n_total} races OK. Log: {LOG_PATH}", flush=True)


if __name__ == "__main__":
    main()
