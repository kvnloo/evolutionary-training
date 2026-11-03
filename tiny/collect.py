"""Status/results of a tiny-gym run, read from the Modal results volume (works while lanes are still running)."""
import argparse
import json

import modal

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
a = ap.parse_args()
vol = modal.Volume.from_name("tiny-gym-results")
for e in sorted(vol.listdir(a.run), key=lambda e: e.path):
    pair = e.path.split("/")[-1]
    try:
        recs = [json.loads(x) for x in b"".join(vol.read_file(f"{a.run}/{pair}/log.jsonl")).decode().split("\n") if x]
    except Exception as ex:  # noqa: BLE001
        print(pair, "no log yet", ex)
        continue
    held = {r["model"]: r for r in recs if r["event"] == "held_out"}
    par = {r["model"]: r for r in recs if r["event"] == "parent_fitness"}
    gens = [r for r in recs if r["event"] == "generation"]
    cands = sum(r["event"] == "candidate" for r in recs)
    print(f"== {pair}: {recs[-1]['event']}, {cands} candidates, {len(gens)} generations")
    for k, r in par.items():
        print(f"   parent {k} fitness {r['fitness']:.3f} (fit acc {r['acc']:.3f})")
    if gens:
        print(f"   best merge fitness {gens[-1]['best_fitness']:.3f}")
    for k, r in held.items():
        parts = ", ".join(f"{s} {r[s]['acc']:.3f} (no-answer {r[s]['no_answer']:.2f})" for s in ("gsm8k", "math500") if s in r)
        print(f"   held-out {k}: {parts}")
