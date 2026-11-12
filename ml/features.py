"""Shared feature engineering for training (train.py) and inference (predict.py)."""
import pandas as pd

BASE = ["cpu_util", "temp_c", "packet_errors", "link_status"]
TYPES = ["server", "switch", "link"]
KEYS = ["scenario_id", "component_id"]
BASELINE_TICKS = 10   # first ticks of a history window are treated as 'normal baseline'


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-component rolling and baseline-deviation features. Uses only past/current ticks
    and the first BASELINE_TICKS of the window, never labels, so there is no leakage."""
    df = df.sort_values(KEYS + ["tick"]).reset_index(drop=True)
    g = df.groupby(KEYS)
    for col in BASE:
        df[f"{col}_roll3"] = g[col].transform(lambda s: s.rolling(3, min_periods=1).mean())
        df[f"{col}_max5"] = g[col].transform(lambda s: s.rolling(5, min_periods=1).max())
    base = (df[df["tick"] < BASELINE_TICKS].groupby(KEYS)[BASE].mean()
            .add_suffix("_base").reset_index())
    df = df.merge(base, on=KEYS, how="left")
    for col in BASE:
        df[f"{col}_dev"] = df[col] - df[f"{col}_base"]
    for t in TYPES:
        df[f"is_{t}"] = (df["component_type"] == t).astype(int)
    return df


def feature_columns() -> list:
    cols = list(BASE)
    for col in BASE:
        cols += [f"{col}_roll3", f"{col}_max5", f"{col}_dev"]
    return cols + [f"is_{t}" for t in TYPES]
