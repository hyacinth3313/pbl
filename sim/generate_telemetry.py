"""
Synthetic telemetry + fault-scenario generator.  ALL DATA IS SYNTHETIC - label every
result produced from it as synthetic in the paper.

Outputs (default folder: data/):
  telemetry.csv   one row per (scenario, tick, component)
  scenarios.json  one entry per scenario with the ground-truth source and impact

Run from the project root:   python -m sim.generate_telemetry --scenarios 200 --seed 42
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sim.topology import (component_type, compute_impact, export_json,
                          monitored_components)

TICKS = 40
ONSET_RANGE = (15, 22)          # fault starts somewhere in this tick range
FAULT_TYPES = ["none", "overheat", "link_failure", "switch_failure"]
FAULT_PROBS = [0.20, 0.35, 0.25, 0.20]

# normal behaviour per component type: cpu mean/sd, temp mean/sd, packet-error rate
NORMAL = {
    "server": dict(cpu=(40, 9), temp=(45, 3), err=1.0),
    "switch": dict(cpu=(25, 6), temp=(40, 2.5), err=1.5),
    "link":   dict(cpu=(0, 0), temp=(0, 0), err=0.7),
}


def _ramp(t, onset, dur):
    return np.clip((t - onset) / dur, 0.0, 1.0)


def simulate_scenario(rng, sid, fault_type, comps, split):
    t = np.arange(TICKS)
    onset = int(rng.integers(*ONSET_RANGE)) if fault_type != "none" else -1
    series = {}
    for c in comps:
        p = NORMAL[component_type(c)]
        series[c] = dict(
            cpu=rng.normal(*p["cpu"], TICKS) if p["cpu"][1] else np.zeros(TICKS),
            temp=rng.normal(*p["temp"], TICKS) if p["temp"][1] else np.zeros(TICKS),
            err=rng.poisson(p["err"], TICKS).astype(float),
            status=np.ones(TICKS),
        )

    # benign load spikes (hard negatives): look odd but are NOT faults
    for c in comps:
        if component_type(c) != "link" and rng.random() < 0.15:
            s0 = int(rng.integers(3, TICKS - 10))
            pulse = ((t >= s0) & (t < s0 + 8)).astype(float)
            series[c]["cpu"] += rng.uniform(25, 40) * pulse
            series[c]["temp"] += rng.uniform(4, 9) * pulse

    meta = dict(scenario_id=sid, fault_type=fault_type, source_id=None,
                onset_tick=onset, split=split, impact={})
    if fault_type != "none":
        pool = {"overheat": [c for c in comps if component_type(c) != "link"],
                "link_failure": [c for c in comps if component_type(c) == "link"],
                "switch_failure": [c for c in comps if component_type(c) == "switch"]}[fault_type]
        src = str(rng.choice(pool))
        impact = compute_impact(src)
        meta.update(source_id=src, impact=impact)
        after = (t >= onset).astype(float)
        r = _ramp(t, onset, 8)
        s = series[src]

        if fault_type == "overheat":
            s["temp"] += rng.uniform(25, 40) * r
            s["cpu"] += 10 * r
            s["err"] += rng.poisson(5 * r + 0.01)
            intensity = 0.35
        elif fault_type == "switch_failure":
            s["cpu"] *= (1 - 0.85 * after)
            s["temp"] -= 5 * after
            s["err"] += rng.poisson(60 * after + 0.01)
            intensity = 1.0
        else:  # link_failure
            s["status"] = 1 - after
            s["err"] += rng.poisson(80 * after + 0.01)
            intensity = 1.0

        # symptoms on the components the fault propagates to
        for c, level in impact.items():
            if c.startswith("svc-") or c not in series:
                continue
            ctype, d = component_type(c), series[c]
            if level == "down":
                d["err"] += rng.poisson(15 * intensity * after + 0.01)
                if ctype == "server":
                    d["cpu"] -= 25 * intensity * after
                if ctype == "link" and intensity == 1.0:
                    d["status"] = 1 - after
            else:  # degraded (rerouted traffic)
                d["err"] += rng.poisson(4 * intensity * after + 0.01)
                if ctype != "link":
                    d["cpu"] += 6 * intensity * after

    rows = []
    for c in comps:
        d, ctype = series[c], component_type(c)
        lvl = meta["impact"].get(c, "none")
        for k in range(TICKS):
            active = onset >= 0 and k >= onset
            rows.append((sid, k, c, ctype,
                         round(float(np.clip(d["cpu"][k], 0, 100)), 2),
                         round(float(max(d["temp"][k], 0)), 2),
                         int(max(d["err"][k], 0)), int(d["status"][k]),
                         fault_type if (active and c == meta["source_id"]) else "none",
                         int(active and c == meta["source_id"]),
                         lvl if active else "none", fault_type, onset, split))
    return rows, meta


def main(n_scenarios, seed, out_dir):
    rng = np.random.default_rng(seed)
    comps = monitored_components()
    ftypes = rng.choice(FAULT_TYPES, size=n_scenarios, p=FAULT_PROBS)
    # split BY SCENARIO (70/30) so no scenario appears in both train and test
    split = np.where(rng.random(n_scenarios) < 0.7, "train", "test")
    rows, metas = [], []
    for sid in range(n_scenarios):
        r, m = simulate_scenario(rng, sid, str(ftypes[sid]), comps, str(split[sid]))
        rows += r
        metas.append(m)

    cols = ["scenario_id", "tick", "component_id", "component_type", "cpu_util",
            "temp_c", "packet_errors", "link_status", "fault_label", "is_source",
            "impact_level", "scenario_fault_type", "onset_tick", "split"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out / "telemetry.csv", index=False)
    (out / "scenarios.json").write_text(json.dumps(metas, indent=1))
    export_json(out / "topology.json")
    print(f"{len(df):,} rows, {n_scenarios} scenarios -> {out}/")
    print(df.drop_duplicates('scenario_id').groupby(['split', 'scenario_fault_type']).size().unstack(0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    main(a.scenarios, a.seed, a.out)
