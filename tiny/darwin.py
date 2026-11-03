"""Darwin merge for tiny same-architecture pairs (base x instruct), ported from scripts/continuous_evolution.py
(Genome, simplified CMA-ES) and scripts/cosmos_qwen3_darwin_merge.py (MRI-Trust Fusion), kept import-light.

Source semantics kept:
- static_term / r_MRI exactly as cosmos_qwen3_darwin_merge.py.
- r_final = tau * r_MRI + (1 - tau) * r_genome, out = (1 - r) * A + r * B.
- CMA-ES: Southpaw's simplified version (mean of the top half, sigma x0.9 on improvement, x1.05 otherwise).

Extension (the source only used rho_b and tau of its 14 dims): r_genome is no longer one global rho_b. It is the
mean of a tensor-class weight (alpha_attn / alpha_ffn / alpha_emb; norms and other tensors use rho_b) and a depth
band weight r0..r5 (six equal bands of layers). gamma, rho_a and lambda_reg stay unused, as in the source.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, fields

FIELDS = ("gamma", "alpha_attn", "alpha_ffn", "alpha_emb", "rho_a", "rho_b", "r0", "r1", "r2", "r3", "r4", "r5",
          "tau", "lambda_reg")


@dataclass
class Genome:
    gamma: float = 0.5
    alpha_attn: float = 0.5
    alpha_ffn: float = 0.5
    alpha_emb: float = 0.5
    rho_a: float = 0.5
    rho_b: float = 0.5
    r0: float = 0.5
    r1: float = 0.5
    r2: float = 0.5
    r3: float = 0.5
    r4: float = 0.5
    r5: float = 0.5
    tau: float = 0.4
    lambda_reg: float = 0.01

    def to_vector(self) -> list[float]:
        return [getattr(self, f.name) for f in fields(self)]

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_vector(cls, v) -> "Genome":
        return cls(*[float(x) for x in v])


class CMAES:
    """Southpaw's simplified CMA-ES (continuous_evolution.py), with a seeded RNG for reproducibility."""

    def __init__(self, genome_size: int = 14, sigma: float = 0.3, pop_size: int = 8, sigma_min: float = 0.05,
                 seed: int = 0):
        self.genome_size, self.sigma, self.pop_size, self.sigma_min = genome_size, sigma, pop_size, sigma_min
        self.mean = [0.5] * genome_size
        self.mean[FIELDS.index("tau")] = 0.4
        self.generation = 0
        self.best_genome: Genome | None = None
        self.best_fitness = float("-inf")
        self.rng = random.Random(seed)

    def ask(self) -> list[Genome]:
        return [Genome.from_vector([max(0.0, min(1.0, self.mean[i] + self.rng.gauss(0, self.sigma)))
                                    for i in range(self.genome_size)]) for _ in range(self.pop_size)]

    def tell(self, candidates: list[Genome], fitnesses: list[float]):
        ranked = sorted(zip(fitnesses, range(len(candidates)), candidates), key=lambda x: -x[0])
        improved = ranked[0][0] > self.best_fitness
        if improved:
            self.best_fitness, self.best_genome = ranked[0][0], ranked[0][2]
        top = ranked[: max(1, len(ranked) // 2)]
        self.mean = [sum(g.to_vector()[i] for _, _, g in top) / len(top) for i in range(self.genome_size)]
        self.sigma = max(self.sigma_min, self.sigma * (0.9 if improved else 1.05))
        self.generation += 1
        return self.best_genome, self.best_fitness

    def state(self) -> dict:
        return {"mean": self.mean, "sigma": self.sigma, "generation": self.generation,
                "best_fitness": self.best_fitness,
                "best_genome": self.best_genome.to_json() if self.best_genome else None}


def static_term(t):
    """Paper's Static: normalized entropy + variance + capped l2-norm (verbatim from the Southpaw merge)."""
    import torch  # noqa: PLC0415
    a = t.float().abs() + 1e-12
    p = a / a.sum()
    H = -(p * p.log()).sum()
    V = t.float().var().sqrt() + 1e-12
    N = torch.clamp(t.float().norm(), max=t.float().norm().item() / 5 + 1e-12)
    return H + V.sqrt() + N.log()


_LAYER = re.compile(r"\.layers\.(\d+)\.")


def tensor_class(name: str) -> str:
    if "embed" in name or name.endswith("lm_head.weight"):
        return "emb"
    if re.search(r"(self_attn|linear_attn|attention)\.", name):
        return "attn"
    if ".mlp." in name:
        return "ffn"
    return "other"


def r_genome(g: Genome, name: str, n_layers: int) -> float:
    cls = {"attn": g.alpha_attn, "ffn": g.alpha_ffn, "emb": g.alpha_emb}.get(tensor_class(name), g.rho_b)
    m = _LAYER.search(name)
    if not m:
        return cls
    band = min(5, int(int(m.group(1)) * 6 / max(1, n_layers)))
    return 0.5 * (cls + getattr(g, f"r{band}"))


def merge_ratios(g: Genome, r_mri: dict[str, float], n_layers: int) -> dict[str, float]:
    return {k: g.tau * rm + (1 - g.tau) * r_genome(g, k, n_layers) for k, rm in r_mri.items()}


if __name__ == "__main__":  # quick self-check without torch
    es = CMAES(seed=1)
    for _ in range(3):
        c = es.ask()
        es.tell(c, [sum(x.to_vector()[:3]) for x in c])
    print(json.dumps(es.state())[:200])
    print(tensor_class("model.layers.3.self_attn.q_proj.weight"), tensor_class("model.layers.3.mlp.up_proj.weight"),
          r_genome(Genome(), "model.layers.20.mlp.up_proj.weight", 24))
