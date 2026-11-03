"""Spawn tiny-gym lanes on Modal (detached: they keep running if groot's client dies). Records call ids."""
import argparse
import json
import time
from pathlib import Path

import modal

STATE = Path.home() / ".z0int" / "state" / "tiny-gym"

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--pairs", required=True, help="comma list, e.g. smollm2-135m,qwen35-0.8b")
ap.add_argument("--gens", type=int, default=6)
ap.add_argument("--pop", type=int, default=8)
a = ap.parse_args()
fn = modal.Function.from_name("tiny-slm-gym", "evolve")
d = STATE / a.run
d.mkdir(parents=True, exist_ok=True)
calls = json.loads((d / "calls.json").read_text()) if (d / "calls.json").exists() else {}
for pair in a.pairs.split(","):
    if pair in calls:
        print("already spawned", pair, calls[pair]["id"])
        continue
    c = fn.spawn(a.run, pair, gens=a.gens, pop=a.pop)
    calls[pair] = {"id": c.object_id, "spawned": time.strftime("%FT%T%z"), "gens": a.gens, "pop": a.pop}
    print("spawned", pair, c.object_id, flush=True)
    (d / "calls.json").write_text(json.dumps(calls, indent=1))
