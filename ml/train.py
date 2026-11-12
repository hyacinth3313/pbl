"""
Train + evaluate:  Random Forest (known fault class) and Isolation Forest (anomaly).
Also evaluates a simple threshold-alert baseline (the 'conventional monitoring' stand-in).
Run from the project root:   python -m ml.train
All numbers come from SYNTHETIC data and must be labelled that way in the paper.
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             precision_recall_fscore_support, roc_auc_score)

from ml.features import add_features, feature_columns

DATA, MODELS, RESULTS = Path("data"), Path("ml/models"), Path("eval/results")
CLASSES = ["none", "overheat", "link_failure", "switch_failure"]


def threshold_baseline(df):
    """Conventional alerting: fixed thresholds on single components."""
    pred = np.full(len(df), "none", dtype=object)
    pred[(df.temp_c > 65).values] = "overheat"
    pred[((df.component_type == "link") & (df.link_status == 0)).values] = "link_failure"
    pred[((df.component_type == "switch") & (df.packet_errors > 30)).values] = "switch_failure"
    return pred


def top1_localization(df, score, truth):
    """Fraction of faulty test scenarios where the highest-scoring component is the true source."""
    d = df[["scenario_id", "component_id"]].copy()
    d["score"] = score
    best = d.groupby(["scenario_id", "component_id"])["score"].max().reset_index()
    pick = best.loc[best.groupby("scenario_id")["score"].idxmax()].set_index("scenario_id")["component_id"]
    hits = [pick[sid] == src for sid, src in truth.items()]
    return float(np.mean(hits)), len(hits)


def prf(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return dict(precision=round(p, 4), recall=round(r, 4), f1=round(f, 4))


def main():
    df = add_features(pd.read_csv(DATA / "telemetry.csv"))
    feats = feature_columns()
    train, test = df[df.split == "train"], df[df.split == "test"]
    assert not set(train.scenario_id) & set(test.scenario_id), "scenario leakage!"
    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {"data": "SYNTHETIC", "train_scenarios": int(train.scenario_id.nunique()),
           "test_scenarios": int(test.scenario_id.nunique())}

    # ---- Random Forest: component-level fault classification -----------------
    rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, n_jobs=-1,
                                class_weight="balanced_subsample", random_state=42)
    rf.fit(train[feats], train.fault_label)
    rf_pred = rf.predict(test[feats])
    base_pred = threshold_baseline(test)
    out["rf_classification"] = classification_report(test.fault_label, rf_pred, labels=CLASSES,
                                                     output_dict=True, zero_division=0)
    out["baseline_classification"] = classification_report(test.fault_label, base_pred, labels=CLASSES,
                                                           output_dict=True, zero_division=0)
    ConfusionMatrixDisplay.from_predictions(test.fault_label, rf_pred, labels=CLASSES,
                                            xticks_rotation=30, colorbar=False)
    plt.title("Random Forest (synthetic test set)")
    plt.tight_layout()
    plt.savefig(RESULTS / "rf_confusion_matrix.png", dpi=150)
    plt.close()

    # ---- Isolation Forest: unsupervised, fitted on fault-free normal rows only
    normal = train[(train.scenario_fault_type == "none") | (train.tick < train.onset_tick)]
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42, n_jobs=-1)
    iso.fit(normal[feats])
    iso_score = -iso.decision_function(test[feats])        # higher = more anomalous
    abnormal = ((test.fault_label != "none") | (test.impact_level != "none")).astype(int)
    out["isolation_forest_detection"] = {
        **prf(abnormal, (iso.predict(test[feats]) == -1).astype(int)),
        "roc_auc": round(float(roc_auc_score(abnormal, iso_score)), 4)}
    out["rf_detection_fault_vs_not"] = prf((test.fault_label != "none").astype(int),
                                           (rf_pred != "none").astype(int))
    out["baseline_detection_fault_vs_not"] = prf((test.fault_label != "none").astype(int),
                                                 (base_pred != "none").astype(int))

    # ---- Fault-source localization (Top-1) -----------------------------------
    scen = {m["scenario_id"]: m for m in json.loads((DATA / "scenarios.json").read_text())}
    truth = {sid: m["source_id"] for sid, m in scen.items()
             if m["split"] == "test" and m["fault_type"] != "none"}
    p_fault = 1 - rf.predict_proba(test[feats])[:, list(rf.classes_).index("none")]
    acc_rf, n = top1_localization(test, p_fault, truth)
    acc_iso, _ = top1_localization(test, iso_score, truth)
    acc_base, _ = top1_localization(test, (base_pred != "none").astype(float), truth)
    out["localization_top1"] = {"n_faulty_test_scenarios": n, "random_forest": round(acc_rf, 4),
                                "isolation_forest": round(acc_iso, 4), "threshold_baseline": round(acc_base, 4),
                                "random_guess": round(1 / df.component_id.nunique(), 4)}

    imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)
    out["rf_top_features"] = {k: round(float(v), 4) for k, v in imp.head(8).items()}

    joblib.dump({"model": rf, "features": feats, "classes": list(rf.classes_)}, MODELS / "rf.joblib")
    joblib.dump({"model": iso, "features": feats}, MODELS / "iforest.joblib")
    (RESULTS / "ml_metrics.json").write_text(json.dumps(out, indent=2))

    f = lambda d: {k: round(d[k], 3) for k in ("precision", "recall", "f1-score")}
    print("\nRandom Forest per class:")
    for c in CLASSES:
        print(f"  {c:15s}", f(out["rf_classification"][c]))
    print("RF macro F1:      ", round(out["rf_classification"]["macro avg"]["f1-score"], 3))
    print("Baseline macro F1:", round(out["baseline_classification"]["macro avg"]["f1-score"], 3))
    print("Isolation Forest: ", out["isolation_forest_detection"])
    print("Detection fault-vs-not  RF:", out["rf_detection_fault_vs_not"], " baseline:", out["baseline_detection_fault_vs_not"])
    print("Top-1 localization:", out["localization_top1"])
    print("Top features:", out["rf_top_features"])


if __name__ == "__main__":
    main()
