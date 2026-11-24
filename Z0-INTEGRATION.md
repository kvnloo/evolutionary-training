# z0 / Evolution Lab integration

This fork is consumed by [kvnloo/evolution-lab](https://github.com/kvnloo/evolution-lab)
through the `z0.training.autoresearch.v1` provider protocol.

**Credit:** the Gym Trainer, Darwin/CMA-ES pipeline, SFT/GRPO training code, data
ingestion, and benchmark machinery are SouthpawIN's implementation. The z0 layer is
only an adapter/orchestrator around those existing surfaces.

## 12 GB smoke arm

Southpaw's canonical configs are optimized for RTX 3090 24 GB cards. The
`qlora-7b-12gb-smoke` config is a deliberately tiny integration canary for a single
12 GB Ampere GPU:

- 4-bit QLoRA
- rank 8
- `q_proj` + `v_proj` only
- batch size 1
- 512-token sequences
- one epoch

It is **not** a quality-training recipe and must not replace the canonical Stage-1
configuration. Its job is to prove that Hermes can drive the real provider path end to
end without OOMing the machine before larger experiments are admitted.

Start with a tiny sample cap:

```bash
python3 scripts/agentic_training_loop.py \
  --train \
  --base-model /path/to/base-model \
  --config qlora-7b-12gb-smoke \
  --max-samples 16
```

Evolution Lab remains responsible for experiment budgets, protected-eval separation,
selection, and promotion. Development evaluation here cannot mint promotion evidence;
protected certification is owned by the pinned z0evals evaluator.
