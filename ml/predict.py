"""
Inference used by Member 2's FastAPI  POST /detect.

    from ml.predict import detect
    result = detect(history_df)

`history_df` = recent telemetry for ALL monitored components, one row per
(tick, component_id), with columns:
    tick, component_id, component_type, cpu_util, temp_c, packet_errors, link_status
Include at least the first 10 ticks of normal behaviour (used as each component's baseline).
"""
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from ml.features import add_features

MODELS = Path(__file__).parent / "models"
RECENT = 5   # ticks used to decide the current state


@lru_cache(maxsize=1)
def _load():
    return joblib.load(MODELS / "rf.joblib"), joblib.load(MODELS / "iforest.joblib")


def _severity(p: float) -> str:
    return "none" if p < 0.2 else "low" if p < 0.5 else "medium" if p < 0.8 else "high"


def detect(history: pd.DataFrame) -> dict:
    rf_b, iso_b = _load()
    df = history.copy()
    if "scenario_id" not in df:
        df["scenario_id"] = 0
    df = add_features(df)
    feats = rf_b["features"]

    proba = rf_b["model"].predict_proba(df[feats])
    classes = rf_b["classes"]
    none_idx = classes.index("none")
    df["p_fault"] = 1 - proba[:, none_idx]
    df["anomaly_score"] = -iso_b["model"].decision_function(df[feats])
    df["cls"] = [classes[i] for i in proba.argmax(axis=1)]

    recent = df[df.tick > df.tick.max() - RECENT]
    comps = []
    for cid, g in recent.groupby("component_id"):
        last = g.sort_values("tick").iloc[-1]
        comps.append({"id": cid, "fault_probability": round(float(g.p_fault.max()), 4),
                      "predicted_class": last.cls,
                      "anomaly_score": round(float(g.anomaly_score.max()), 4)})
    comps.sort(key=lambda c: c["fault_probability"], reverse=True)
    top = comps[0]
    faulty = top["fault_probability"] >= 0.5
    return {
        "fault_detected": faulty,
        "suspected_source": top["id"] if faulty else None,
        "fault_class": top["predicted_class"] if faulty else "none",
        "confidence": top["fault_probability"],
        "severity": _severity(top["fault_probability"]),
        "components": comps,
    }


if __name__ == "__main__":      # demo: replay a held-out test scenario up to 10 ticks after onset
    import json
    data = pd.read_csv("data/telemetry.csv")
    meta = [m for m in json.loads(Path("data/scenarios.json").read_text())
            if m["split"] == "test" and m["fault_type"] != "none"]
    for m in meta[:5]:
        h = data[(data.scenario_id == m["scenario_id"]) & (data.tick <= m["onset_tick"] + 10)]
        r = detect(h.drop(columns=["scenario_id"]))
        print(f"scenario {m['scenario_id']:3d} true={m['source_id']:14s}({m['fault_type']:14s}) -> "
              f"pred={r['suspected_source']} ({r['fault_class']}, conf {r['confidence']}, {r['severity']})")
