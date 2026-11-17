# Car or Driver?

Can a model identify an F1 driver from a single lap of telemetry, or does it just learn which car they're in?

Lap telemetry (speed, throttle, brake, RPM, gear) mixes two signals: the driver's style and the car's characteristics. Teammates share a car, and most drivers stay at one team for years, so a classifier can score really well just by recognizing the car. The real test is drivers who changed teams, e.g. Hamilton moving from Mercedes to Ferrari in 2025.

The goal is to learn driver embeddings that are invariant to the car, using conditional adversarial training (an adversary tries to predict the car-season from the embedding + driver, and the encoder is trained to fool it). This repo currently has the data pipeline, baselines, and evaluation; the adversarial part is in progress.

Project for Stanford CS230 (Deep Learning), Fall 2026.

## Data

Race telemetry for 2018-2025 via [FastF1](https://github.com/theOehrly/Fast-F1). The timing archive needs an **F1 TV subscription**, so the data isn't included here.

- 171 races downloaded, 131k laps after cleaning (green flag only, no in/out laps, no lap 1, within 107% of the fastest lap, dry tyres only)
- each lap resampled onto 512 points along normalized distance -> `(5, 512)` array
- splits are by race, never by lap: train 2018-2024, val = every 4th 2024 race, test = all of 2025

## Model

FCN (Wang et al., 2017): three Conv1d/BatchNorm/ReLU blocks (128/256/128 filters), global average pooling to a 128-d embedding `z`, then a linear driver classifier. ~274k params, trains in ~20 min on an M3 Pro (MPS).

## Results so far

| run | val acc | val teammate | 2025 same team | 2025 switchers | switchers teammate |
|---|---|---|---|---|---|
| FCN, all 5 channels | 97.3 | 98.1 | 94.6 | 52.6 | 74.9 |
| FCN, no RPM/gear | 74.2 | 88.8 | 67.8 | 40.8 | 57.9 |

The model does great on drivers who stayed at their team and drops to ~50% on drivers who switched. That gap is the car shortcut. Hamilton's 2025 Ferrari laps get predicted as **Sainz** (who drove that Ferrari seat before him) 49% of the time and as Hamilton only 22%.

A linear probe on `z` that tries to tell a driver's 2022 car from their 2023 car gets 84% (vs 72.5% for an untrained encoder), so the embedding is carrying car information. Dropping RPM and gear by hand cuts that to 67% but hurts accuracy everywhere, and Hamilton then shows up as Leclerc 53% of the time, so the shortcut is still there.

## Layout

```
src/
  download.py     pull races into the FastF1 cache (handles the API rate limit)
  preprocess.py   clean laps, resample to (5, 512), save per-season npz + parquet
  data.py         splits, normalization, Dataset/DataLoaders
  models.py       FCN
  train.py        training loop, keeps best-val checkpoint
  evaluate.py     val/test metrics, team-switcher tiers, teammate accuracy
  probe.py        linear probe for car leakage in the embedding
notebooks/
  01_eda.ipynb    first look: Hamilton vs Russell at Bahrain 2024
figures/
```

## Running it

```bash
pip install -r requirements.txt
python -m fastf1 auth f1tv --authenticate     # needs F1 TV
python src/download.py --start 2018 --end 2025  # ~5 hrs, rate limited
python src/preprocess.py
python src/train.py --tag naive_fcn
python src/train.py --tag no_rpm_gear --channels Speed Throttle Brake
python src/evaluate.py naive_fcn no_rpm_gear
python src/probe.py naive_fcn no_rpm_gear
```

## Next

- conditional adversarial training (gradient reversal on `[z, driver]` -> car-season), plus plain DANN as an ablation
- sweep the adversarial weight, rolling year-by-year evaluation folds, multiple seeds
- embedding visualizations for the team switchers
