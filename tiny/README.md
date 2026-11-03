# Tiny SLM gym

Can sub-1B models be made genuinely useful? This directory runs Southpaw's Gym Trainer method (Darwin
evolution with MRI-Trust Fusion and the simplified CMA-ES, from `scripts/`) on tiny base × instruct pairs, scored on
held-out math. It pairs with the weight-space programming harness in z0intelligence
(`benchmarks/weight_space`, branch `feat/weight-space-evals`), where frontier models edit the raw weights.

## Ladder (both ends in parallel)

| Lane | Pair (parent A = base, parent B = instruct) |
|---|---|
| smallest | `smollm2-135m`: SmolLM2-135M × SmolLM2-135M-Instruct |
| small | `gemma3-270m`: gemma-3-270m × gemma-3-270m-it (unsloth ungated mirrors) |
| large | `qwen3-0.6b`: Qwen3-0.6B-Base × Qwen3-0.6B |
| largest under 1B | `qwen35-0.8b`: Qwen3.5-0.8B-Base × Qwen3.5-0.8B |

All revisions are pinned in `modal_gym.py`.

## Protocol

- Fitness: 64 fixed GSM8K **train** problems. Fitness = accuracy + 0.1 × exp(mean token log-prob of the reference
  solution). The log-prob term only breaks ties; it matters for 135M models that solve almost nothing.
- Held-out: GSM8K test (250, seed 0) + MATH-500 (100, seed 0), pinned dataset revisions, never used for fitness.
- Every variant is prompted through the instruct parent's chat template (Qwen thinking off), greedy, 512 new tokens.
  The answer is the last `\boxed{}`, with a last-number fallback for GSM8K; graded with math-verify.
- Genome: Southpaw's 14 dims. The source merge used only `rho_b` and `tau`. Here, `r_genome` is the mean of a
  tensor-class weight (`alpha_attn` / `alpha_ffn` / `alpha_emb`) and a depth-band weight (`r0`..`r5`). `gamma`,
  `rho_a` and `lambda_reg` stay unused, as in the source. See `darwin.py`.
- The source's known `CMAES.tell` crash on tied scores (it sorted non-orderable Genome objects) is fixed by ranking on
  (fitness, index).

## Run

```bash
modal deploy tiny/modal_gym.py
python tiny/launch.py --run <run-id> --pairs smollm2-135m,qwen35-0.8b   # spawned detached on Modal
python tiny/collect.py --run <run-id>                                     # progress and held-out results
```

A lane runs end to end in one Modal GPU container: parents, CMA-ES, best merge, held-out scoring. It writes
`log.jsonl`, `held_<model>.jsonl` (per item), `summary.json` and `best_merge/` to the `tiny-gym-results` volume.
groot only launches and collects.

## Profiles

`tiny-trainer` runs lanes and reports; `tiny-evaluator` independently checks claimed gains (paired, per item). Both
were created fresh: no copied credentials, memories or schedules, and no schedules are armed.

## Next arms

1. Frontier weight-space editing starting from the best merge (z0int SLM pilot, `WSP_BACKEND=modal`).
2. Short SFT/LoRA on GSM8K train, the Southpaw loop B, compared at equal held-out items.
