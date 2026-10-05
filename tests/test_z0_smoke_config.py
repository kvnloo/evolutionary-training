#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "agentic_training_loop.py"

spec = importlib.util.spec_from_file_location("agentic_training_loop", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

cfg = module.TRAINING_CONFIGS["qlora-7b-12gb-smoke"]
assert cfg["method"] == "qlora"
assert cfg["bits"] == 4
assert cfg["batch_size"] == 1
assert cfg["max_seq_len"] <= 512
assert cfg["lora_r"] <= 8
assert cfg["epochs"] == 1
assert set(cfg["target_modules"]) <= {"q_proj", "v_proj"}
print("ok: qlora-7b-12gb-smoke is bounded for z0 canary use")
