# Member 1: Data + ML

Run from the project root (needs `pip install -r requirements.txt`):

```bash
python -m sim.topology              # prints example impact sets, writes data/topology.json
python -m sim.generate_telemetry    # data/telemetry.csv + data/scenarios.json (synthetic)
python -m ml.train                  # trains RF + Isolation Forest, writes eval/results/
python -m ml.predict                # demo: replays held-out scenarios through detect()
```

| File | Purpose |
|---|---|
| `sim/topology.py` | Component IDs, real-size 3D positions, links, services, `compute_impact()` ground truth |
| `sim/generate_telemetry.py` | Synthetic telemetry, fault injection, train/test split by scenario |
| `ml/features.py` | Rolling and baseline-deviation features (shared by train and predict) |
| `ml/train.py` | Random Forest, Isolation Forest, threshold baseline, Top-1 localization |
| `ml/predict.py` | `detect(history_df)` for Member 2's `POST /detect` |

All results come from SYNTHETIC data. Label them that way in the paper.
