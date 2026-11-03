"""Tiny SLM gym on Modal: Southpaw Darwin evolution (base x instruct) for sub-1B models, scored on held-out math.

    modal deploy tiny/modal_gym.py
    python tiny/launch.py --run 20261103-a --pairs smollm2-135m,qwen35-0.8b       # spawn lanes (detached)
    python tiny/collect.py --run 20261103-a                                       # fetch results

Each lane runs entirely in one GPU container, so groot only launches and collects:
  1. score both parents on the fitness set (GSM8K train, 64 fixed problems) and the held-out set
  2. CMA-ES over the 14-dim Darwin genome, fitness = accuracy + 0.1 * exp(mean token log-prob of the reference
     solution); the log-prob term only breaks ties between equally accurate merges
  3. score the best merge on the held-out set; save its weights and a full log

Held-out = GSM8K test (250, seed 0) + MATH-500 (100, seed 0), pinned dataset revisions; never used for fitness.
All variants are prompted through the instruct parent's chat template (Qwen thinking off), greedy, 512 new tokens.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP = "tiny-slm-gym"
GPU = os.environ.get("TINY_GPU", "L4")

PAIRS = {
    "smollm2-135m": {"a": ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"),
                     "b": ("HuggingFaceTB/SmolLM2-135M-Instruct", "12fd25f77366fa6b3b4b768ec3050bf629380bac")},
    "gemma3-270m": {"a": ("unsloth/gemma-3-270m", "edcf1fe394ee1168e7c5538afd104cfb1f7947e6"),
                    "b": ("unsloth/gemma-3-270m-it", "23cf460f6bb16954176b3ddcc8d4f250501458a9")},
    "qwen3-0.6b": {"a": ("Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"),
                   "b": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca")},
    "qwen35-0.8b": {"a": ("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"),
                    "b": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")},
}
DATA = {"gsm8k": ("openai/gsm8k", None), "math500": ("HuggingFaceH4/MATH-500", None)}  # revs pinned at run start
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
FIT_N, GSM_TEST_N, MATH_N, MAX_NEW = 64, 250, 100, 512

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.0", "transformers==5.20.2", "accelerate==1.15.0", "flash-linear-attention==0.6.0",
                 "math-verify==0.9.0", "safetensors", "numpy", "huggingface_hub", "pyarrow")
    .env({"HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(str(HERE / "darwin.py"), "/root/darwin.py")
)
hf_vol = modal.Volume.from_name("tiny-gym-hf", create_if_missing=True)
res_vol = modal.Volume.from_name("tiny-gym-results", create_if_missing=True)
app = modal.App(APP)


# ------------------------------------------------------------------ data and grading
def _pin(api_rev: dict) -> dict:
    from huggingface_hub import HfApi  # noqa: PLC0415
    api = HfApi()
    return {k: api_rev.get(k) or api.dataset_info(repo).sha for k, (repo, _) in DATA.items()}


def load_items(revs: dict) -> tuple[list[dict], list[dict]]:
    import pyarrow.parquet as pq  # noqa: PLC0415
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    tr = pq.read_table(hf_hub_download("openai/gsm8k", "main/train-00000-of-00001.parquet", repo_type="dataset",
                                       revision=revs["gsm8k"])).to_pylist()
    te = pq.read_table(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset",
                                       revision=revs["gsm8k"])).to_pylist()
    ans = lambda r: r["answer"].split("####")[-1].strip().replace(",", "")  # noqa: E731
    sol = lambda r: re.sub(r"<<[^>]*>>", "", r["answer"].split("####")[0]).strip()  # noqa: E731
    fit = [{"id": f"gsm8k_train:{i}", "source": "gsm8k", "problem": tr[i]["question"], "answer": ans(tr[i]),
            "solution": sol(tr[i])} for i in sorted(random.Random(0).sample(range(len(tr)), FIT_N))]
    held = [{"id": f"gsm8k_test:{i}", "source": "gsm8k", "problem": te[i]["question"], "answer": ans(te[i])}
            for i in sorted(random.Random(0).sample(range(len(te)), GSM_TEST_N))]
    p = hf_hub_download("HuggingFaceH4/MATH-500", "test.jsonl", repo_type="dataset", revision=revs["math500"])
    m = [json.loads(x) for x in Path(p).read_text().split("\n") if x.strip()]
    held += [{"id": "math500:" + r["unique_id"], "source": "math500", "problem": r["problem"], "answer": r["answer"]}
             for r in random.Random(0).sample(m, MATH_N)]
    return fit, held


_BOX = re.compile(r"\\(?:boxed|fbox)\s*\{")


def last_boxed(text: str) -> str | None:
    ms = list(_BOX.finditer(text.split("</think>")[-1]))
    if not ms:
        return None
    t = text.split("</think>")[-1]
    i = ms[-1].end()
    depth, j = 1, i
    while j < len(t) and depth:
        depth += {"{": 1, "}": -1}.get(t[j], 0)
        j += 1
    return t[i:j - 1].strip() if depth == 0 else None


def extract(text: str, source: str) -> str | None:
    b = last_boxed(text)
    if b is not None or source != "gsm8k":
        return b
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def is_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        from math_verify import parse, verify  # noqa: PLC0415
        g, p = parse("\\boxed{" + gold + "}"), parse("\\boxed{" + pred + "}")
        if g and p and verify(g, p):
            return True
    except Exception:  # noqa: BLE001, S110 - exact normalized comparison below
        pass
    norm = lambda s: re.sub(r"[\s$]|\\[!,;]|\\text\{[^}]*\}|\\left|\\right", "", s).rstrip(".")  # noqa: E731
    return norm(pred) == norm(gold)


# ------------------------------------------------------------------ model side
class Lane:
    def __init__(self, pair: str):
        import torch  # noqa: PLC0415
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        from safetensors.torch import load_file  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
        import darwin  # noqa: PLC0415
        self.torch, self.D = torch, darwin
        pat = ["*.json", "*.safetensors", "*.jinja", "*.model", "*.txt"]
        self.dirs = {k: Path(snapshot_download(PAIRS[pair][k][0], revision=PAIRS[pair][k][1], allow_patterns=pat))
                     for k in ("a", "b")}
        hf_vol.commit()
        self.tok = AutoTokenizer.from_pretrained(str(self.dirs["b"]))
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(str(self.dirs["b"]), dtype=torch.bfloat16).cuda().eval()
        cfg = json.loads((self.dirs["b"] / "config.json").read_text())
        self.n_layers = int(cfg.get("num_hidden_layers") or cfg.get("text_config", {}).get("num_hidden_layers"))
        self.qwen = "qwen" in pair

        def sd(d):
            out = {}
            for f in sorted(d.glob("*.safetensors")):
                out.update(load_file(str(f), device="cuda"))
            return out
        A, B = sd(self.dirs["a"]), sd(self.dirs["b"])
        self.params = dict(self.model.named_parameters())
        # map checkpoint keys onto the instruct model's parameter names (identical for these pairs)
        def ck(p):  # parameter name -> checkpoint key present in both parents
            for c in (p, p.replace("model.", "model.language_model.", 1), "model." + p):
                if c in A and c in B and A[c].shape == B[c].shape == self.params[p].shape:
                    return c
            return None
        cmap = {p: ck(p) for p in self.params}
        self.keys = [p for p, c in cmap.items() if c]
        self.missing = sorted(p for p, c in cmap.items() if not c)
        frac = sum(self.params[p].numel() for p in self.keys) / sum(v.numel() for v in self.params.values())
        assert frac > 0.95, f"only {frac:.1%} of parameters map onto both parents: {self.missing[:5]}"
        self.A = {p: A[cmap[p]].float() for p in self.keys}
        self.B = {p: B[cmap[p]].float() for p in self.keys}
        del A, B
        self.r_mri = {}
        for k in self.keys:
            sa, sb = darwin.static_term(self.A[k]), darwin.static_term(self.B[k])
            self.r_mri[k] = float(sb / (sa + sb))

    def set_weights(self, which):
        """which: 'a', 'b', or a darwin.Genome."""
        with self.torch.no_grad():
            if which in ("a", "b"):
                src = self.A if which == "a" else self.B
                for k in self.keys:
                    self.params[k].copy_(src[k].to(self.params[k].dtype))
                return None
            r = self.D.merge_ratios(which, self.r_mri, self.n_layers)
            for k in self.keys:
                self.params[k].copy_(((1 - r[k]) * self.A[k] + r[k] * self.B[k]).to(self.params[k].dtype))
            return r

    def prompt(self, problem: str) -> str:
        kw = {"enable_thinking": False} if self.qwen else {}
        return self.tok.apply_chat_template([{"role": "user", "content": problem + "\n" + INSTRUCTION}],
                                            tokenize=False, add_generation_prompt=True, **kw)

    def generate(self, items: list[dict], bs: int = 64) -> list[dict]:
        rows = []
        order = sorted(range(len(items)), key=lambda i: len(items[i]["problem"]))
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            enc = self.tok([self.prompt(items[i]["problem"]) for i in idx], return_tensors="pt", padding=True,
                           add_special_tokens=False).to("cuda")
            with self.torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            for j, i in enumerate(idx):
                gen = out[j, enc["input_ids"].shape[1]:]
                text = self.tok.decode(gen, skip_special_tokens=True)
                pred = extract(text, items[i]["source"])
                rows.append({"id": items[i]["id"], "source": items[i]["source"], "pred": pred,
                             "gold": items[i]["answer"], "ok": is_correct(pred, items[i]["answer"]),
                             "tokens": int((gen != self.tok.pad_token_id).sum()), "text": text[-1500:]})
        return sorted(rows, key=lambda r: r["id"])

    def gold_logprob(self, items: list[dict], bs: int = 4) -> float:
        tot, n = 0.0, 0
        self.tok.padding_side = "right"
        try:
            for s in range(0, len(items), bs):
                batch = items[s:s + bs]
                ps = [self.prompt(it["problem"]) for it in batch]
                full = [p + it["solution"] + f"\nThe answer is \\boxed{{{it['answer']}}}." for p, it in zip(ps, batch)]
                enc = self.tok(full, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                plen = [len(self.tok(p, add_special_tokens=False)["input_ids"]) for p in ps]
                with self.torch.no_grad():
                    logits = self.model(**enc).logits  # bf16; per-sample float slices keep 248k-vocab models small
                    for j, pl in enumerate(plen):
                        end = int(enc["attention_mask"][j].sum())
                        lg = logits[j, pl - 1:end - 1].float()
                        tgt = enc["input_ids"][j, pl:end]
                        tot -= float(self.torch.nn.functional.cross_entropy(lg, tgt, reduction="sum"))
                        n += int(tgt.numel())
                    del logits
        finally:
            self.tok.padding_side = "left"
        return tot / max(1, n)


def _summ(rows: list[dict]) -> dict:
    out = {"n": len(rows), "acc": sum(r["ok"] for r in rows) / max(1, len(rows))}
    for src in sorted({r["source"] for r in rows}):
        s = [r for r in rows if r["source"] == src]
        out[src] = {"n": len(s), "acc": sum(r["ok"] for r in s) / len(s),
                    "no_answer": sum(r["pred"] is None for r in s) / len(s),
                    "mean_tokens": sum(r["tokens"] for r in s) / len(s)}
    return out


@app.function(image=image, gpu=GPU, volumes={"/hf": hf_vol, "/results": res_vol}, timeout=8 * 3600, memory=32768)
def evolve(run: str, pair: str, gens: int = 6, pop: int = 8, seed: int = 0, revs: dict | None = None) -> dict:
    import darwin  # noqa: PLC0415
    root = Path("/results") / run / pair
    root.mkdir(parents=True, exist_ok=True)
    log = root / "log.jsonl"

    def emit(rec):
        rec["t"] = time.time()
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        res_vol.commit()
        print(json.dumps(rec)[:400], flush=True)

    revs = _pin(revs or {})
    fit_items, held = load_items(revs)
    t0 = time.time()
    lane = Lane(pair)
    emit({"event": "start", "pair": pair, "parents": PAIRS[pair], "data_revs": revs, "gpu": GPU,
          "n_layers": lane.n_layers, "merged_tensors": len(lane.keys), "unmerged": lane.missing[:10],
          "fit_n": len(fit_items), "held_n": len(held), "gens": gens, "pop": pop, "seed": seed})

    def fitness(tag):
        rows = lane.generate(fit_items)
        acc = sum(r["ok"] for r in rows) / len(rows)
        lp = lane.gold_logprob(fit_items)
        return acc + 0.1 * math.exp(lp), acc, lp

    def held_out(tag):
        rows = lane.generate(held)
        (root / f"held_{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        s = _summ(rows)
        emit({"event": "held_out", "model": tag, **s})
        return s

    summary = {"pair": pair, "run": run, "parents": {}, "generations": []}
    for k in ("a", "b"):
        lane.set_weights(k)
        f, acc, lp = fitness(k)
        emit({"event": "parent_fitness", "model": k, "fitness": f, "acc": acc, "gold_lp": lp})
        summary["parents"][k] = {"fitness": f, "fit_acc": acc, "gold_lp": lp, "held": held_out(k)}

    es = darwin.CMAES(pop_size=pop, seed=seed)
    for g in range(gens):
        cands = es.ask()
        fits = []
        for i, c in enumerate(cands):
            lane.set_weights(c)
            f, acc, lp = fitness(f"g{g}c{i}")
            fits.append(f)
            emit({"event": "candidate", "gen": g, "cand": i, "fitness": f, "acc": acc, "gold_lp": lp,
                  "genome": c.to_json()})
        best, bf = es.tell(cands, fits)
        st = es.state()
        (root / "cmaes_state.json").write_text(json.dumps(st))
        emit({"event": "generation", "gen": g, "best_fitness": bf, "gen_best": max(fits), "sigma": st["sigma"]})
        summary["generations"].append({"gen": g, "best_fitness": bf, "gen_best": max(fits), "sigma": st["sigma"]})

    lane.set_weights(es.best_genome)
    summary["best"] = {"genome": es.best_genome.to_json(), "fitness": es.best_fitness, "held": held_out("best")}
    from safetensors.torch import save_file  # noqa: PLC0415
    best_dir = root / "best_merge"
    best_dir.mkdir(exist_ok=True)
    save_file({k: v.detach().to("cpu").contiguous() for k, v in lane.model.state_dict().items()},
              str(best_dir / "model.safetensors"), metadata={"format": "pt"})
    for f in lane.dirs["b"].iterdir():
        if f.suffix != ".safetensors" and f.is_file():
            (best_dir / f.name).write_bytes(f.read_bytes())
    (best_dir / "genome.json").write_text(json.dumps(es.best_genome.to_json()))
    summary["wall_s"] = round(time.time() - t0, 1)
    (root / "summary.json").write_text(json.dumps(summary, indent=1))
    emit({"event": "done", "wall_s": summary["wall_s"]})
    return summary
