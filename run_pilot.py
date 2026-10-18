"""Z-FMN on RAGTruth.

For each response written by an open model:
  1. score it with that same model (token log-probs and entropies, teacher forcing),
  2. draw N fresh samples for the same prompt and score them too,
  3. check every answer sentence with an NLI model against the source and against the samples,
  4. build TFMNs, pool evidence into hybrid Z-numbers,
  5. map RAGTruth's human hallucination spans onto nodes and edges,
then compare how well each confidence measure detects hallucinated nodes, edges and answers (AUROC).

Stages (--stage): generate (needs the LLM), nli (needs the NLI model), evaluate (CPU), or all.
Each stage is cached in --out, and --reuse-dir picks up generations and NLI results from earlier runs (for
example a previous Kaggle version attached as input), so a long study can be split across sessions.

Example (Kaggle / Colab GPU):
  python run_pilot.py --model-label mistral-7B-instruct --limit 50 --n-samples 10 --load-in-4bit
"""

import argparse
import csv
import glob
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zfmn.data import OPEN_MODELS, load_ragtruth  # noqa: E402


def read_jsonl(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {d["id"]: d for d in map(json.loads, f)}


def reuse_cached(args, rows, path, ok=lambda d: True):
    """Copy records for these rows from same-named files under --reuse-dir into path."""
    if not args.reuse_dir:
        return
    name = os.path.basename(path)
    want = {r["id"] for r in rows} - set(read_jsonl(path))
    found = {}
    for p in glob.glob(os.path.join(args.reuse_dir, "**", name), recursive=True):
        if os.path.abspath(p) == os.path.abspath(path):
            continue
        for k, d in read_jsonl(p).items():
            if k in want and k not in found and ok(d):
                found[k] = d
    if found:
        with open(path, "a") as f:
            for d in found.values():
                f.write(json.dumps(d) + "\n")
    print(f"[reuse] {len(found)} earlier records reused from {name} files", flush=True)


def stage_generate(args, rows, path):
    from zfmn.lm import load_model, sample_responses, score_response

    reuse_cached(args, rows, path, ok=lambda d: len(d["samples"]) == args.n_samples)
    done = read_jsonl(path)
    todo = [r for r in rows if r["id"] not in done]
    if not todo:
        return
    t0 = time.time()
    model, tok = load_model(args.hf_model, load_in_4bit=args.load_in_4bit)
    with open(path, "a") as f:
        for i, r in enumerate(todo, 1):
            if args.time_budget_hours and time.time() - t0 > args.time_budget_hours * 3600:
                print(f"[generate] time budget reached; {len(todo) - i + 1} left for the next session", flush=True)
                break
            target = score_response(model, tok, r["prompt"], r["response"], args.max_length)
            texts = sample_responses(model, tok, r["prompt"], args.n_samples,
                                     args.temperature, args.max_new_tokens)
            samples = [{"text": t, "tokens": score_response(model, tok, r["prompt"], t, args.max_length)}
                       for t in texts]
            f.write(json.dumps({"id": r["id"], "target_tokens": target, "samples": samples}) + "\n")
            f.flush()
            print(f"[generate] {i}/{len(todo)} response {r['id']}", flush=True)


def stage_nli(args, rows, gen_path, nli_path):
    from zfmn.grounding import NLI, nli_record
    from zfmn.network import get_nlp

    gen = read_jsonl(gen_path)
    reuse_cached(args, [r for r in rows if r["id"] in gen], nli_path)
    done = read_jsonl(nli_path)
    todo = [r for r in rows if r["id"] in gen and r["id"] not in done]
    if not todo:
        return
    nli, nlp = NLI(args.nli_model, batch_size=args.nli_batch), get_nlp()
    with open(nli_path, "a") as f:
        for i, r in enumerate(todo, 1):
            rec = nli_record(r["response"], r["source_text"], [s["text"] for s in gen[r["id"]]["samples"]], nlp, nli)
            f.write(json.dumps({"id": r["id"], **rec}) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"[nli] {i}/{len(todo)}", flush=True)


def response_record(r, g, recs, nli):
    """One row per answer: does it contain any hallucination, and answer-level scores."""
    lps = [t["logprob"] for t in g["target_tokens"]]
    out = {"response_id": r["id"], "task_type": r["task_type"], "wrong": int(bool(r["spans"])),
           "types": ";".join(sorted(set(r["span_types"]))), "n_tokens": len(lps),
           "resp_neg_mean_logprob": -sum(lps) / len(lps) if lps else None}
    if nli:
        sc = [s for s in nli["selfcheck"] if s is not None]
        out["resp_selfcheck_nli"] = sum(sc) / len(sc) if sc else None
        out["resp_nli_contra_max"] = max((s[2] for s in nli["src"]), default=None)
        out["resp_nli_not_entailed"] = 1 - min((s[0] for s in nli["src"]), default=1.0)
    for kind in ("node", "edge"):
        xs = [x for x in recs if x["kind"] == kind]
        for name, f in (("P", lambda x: 1 - x["sl_P"]), ("u", lambda x: x.get("ctx_u")),
                        ("nli_fused_P", lambda x: None if x.get("nli_fused_P") is None else 1 - x["nli_fused_P"]),
                        ("tri_fused_P", lambda x: None if x.get("tri_fused_P") is None else 1 - x["tri_fused_P"])):
            v = [f(x) for x in xs if f(x) is not None]
            out[f"resp_{kind}_{name}_max"] = max(v) if v else None
            out[f"resp_{kind}_{name}_mean"] = sum(v) / len(v) if v else None
    return out


def stage_evaluate(args, rows, gen_path, nli_path):
    from zfmn.advanced import advanced_records
    from zfmn.evidence import SCORERS, element_records
    from zfmn.grounding import element_nli, fused_fields
    from zfmn.metrics import clustered_auroc
    from zfmn.network import build_tfmn

    scorers = dict(SCORERS)
    scorers.update({
        "uncertainty_u (unsupported)": lambda x: x["ctx_u"],
        "disbelief_d (contradicted)": lambda x: x["ctx_d"],
        "source_network_only": lambda x: 1 - x["src_P"],
        "fused_model+source": lambda x: 1 - x["fused_P"],
        "selfcheck_nli (baseline)": lambda x: x.get("selfcheck"),
        "nli_source_contradiction": lambda x: x.get("nli_c"),
        "nli_source_not_entailed": lambda x: None if x.get("nli_e") is None else 1 - x["nli_e"],
        "z_fmn+nli_fused_P": lambda x: None if x.get("nli_fused_P") is None else 1 - x["nli_fused_P"],
        "z_fmn+nli_fused_d": lambda x: x.get("nli_fused_d"),
        "z_fmn+nli+selfcheck_P": lambda x: None if x.get("tri_fused_P") is None else 1 - x["tri_fused_P"],
    })

    gen = read_jsonl(gen_path)
    nli_all = read_jsonl(nli_path)
    all_recs, resp_recs = [], []
    for r in rows:
        g = gen.get(r["id"])
        if g is None:
            continue
        target = build_tfmn(r["response"], g["target_tokens"], args.max_distance)
        samples = [build_tfmn(s["text"], s["tokens"], args.max_distance) for s in g["samples"]]
        source = build_tfmn(r["source_text"], None, args.max_distance)
        adv = advanced_records(target, samples, r["spans"], W=args.prior_weight,
                               span_types=r["span_types"], source=source)
        nli = nli_all.get(r["id"])
        recs = []
        for rec, extra in zip(element_records(target, samples, r["spans"], W=args.prior_weight), adv):
            extra = {k: v for k, v in extra.items() if k not in ("kind", "key", "wrong")}
            extra["types"] = ";".join(extra["types"])
            extra.pop("z", None)
            rec.update(extra)
            if nli:
                if rec["kind"] == "node":
                    spans = [o["span"] for o in target["nodes"][rec["key"]]]
                else:
                    spans = [sp for o in target["edges"][tuple(rec["key"].split("|"))] for sp in o["spans"]]
                en = element_nli(spans, nli)
                if en:
                    rec.update(en)
                    rec.update(fused_fields(rec["sl_b"], rec["sl_u"], en["nli_e"], en["nli_c"],
                                            args.nli_strength, args.prior_weight,
                                            selfcheck=en["selfcheck"], n_samples=args.n_samples))
            rec["response_id"] = r["id"]
            rec["task_type"] = r["task_type"]
            recs.append(rec)
        all_recs += recs
        resp_recs.append(response_record(r, g, recs, nli))

    os.makedirs(args.out, exist_ok=True)
    for name, data in (("elements.csv", all_recs), ("responses.csv", resp_recs)):
        keys = list(dict.fromkeys(k for d in data for k in d))
        with open(os.path.join(args.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(data)

    summary = []
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    subsets = {
        "all": lambda x: True,
        "conflict": lambda x: not x["wrong"] or "Conflict" in x["types"],
        "baseless": lambda x: not x["wrong"] or "Baseless" in x["types"],
    }
    lines = ["| Element | Hallucination type | Measure | AUROC | 95% CI | n (wrong / total) |",
             "| --- | --- | --- | --- | --- | --- |"]
    for kind, (sub, keep) in [(k, kv) for k in ("node", "edge") for kv in subsets.items()]:
        recs = [x for x in all_recs if x["kind"] == kind and keep(x)]
        for name, fn in scorers.items():
            groups = {}
            for x in recs:
                try:
                    s = fn(x)
                except KeyError:
                    s = None
                if s is None:
                    continue
                gs, gl = groups.setdefault(x["response_id"], ([], []))
                gs.append(s)
                gl.append(x["wrong"])
            groups = list(groups.values())
            n = sum(len(g[1]) for g in groups)
            pos = sum(sum(g[1]) for g in groups)
            if not n or pos in (0, n):
                continue
            a, lo, hi = clustered_auroc(groups, n_boot=args.n_boot)
            lines.append(f"| {kind} | {sub} | {name} | {fmt(a)} | [{fmt(lo)}, {fmt(hi)}] | {pos} / {n} |")
            summary.append({"kind": kind, "subset": sub, "measure": name, "auroc": a, "ci_low": lo, "ci_high": hi,
                            "n_wrong": pos, "n": n})
    # answer level: does the answer contain any hallucination? Needs answers without hallucinations too.
    labels = [x["wrong"] for x in resp_recs]
    if 0 < sum(labels) < len(labels):
        lines += ["", "| Answer-level measure | AUROC | 95% CI | n (with hallucination / total) |",
                  "| --- | --- | --- | --- |"]
        for k in [k for k in resp_recs[0] if k.startswith("resp_")]:
            pairs = [(x[k], x["wrong"]) for x in resp_recs if x.get(k) is not None]
            if len({lab for _, lab in pairs}) < 2:
                continue
            a, lo, hi = clustered_auroc([([s], [lab]) for s, lab in pairs], n_boot=args.n_boot)
            npos = sum(lab for _, lab in pairs)
            lines.append(f"| {k} | {fmt(a)} | [{fmt(lo)}, {fmt(hi)}] | {npos} / {len(pairs)} |")
            summary.append({"kind": "answer", "subset": "all", "measure": k, "auroc": a, "ci_low": lo,
                            "ci_high": hi, "n_wrong": npos, "n": len(pairs)})
    table = "\n".join(lines)
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write(f"# Z-FMN on RAGTruth\n\nModel: {args.hf_model}; tasks: {args.tasks}; answers: {len(resp_recs)}; "
                f"samples per prompt: {args.n_samples}; NLI: {args.nli_model if nli_all else 'not run'}\n\n")
        f.write(table + "\n")
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(table)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-label", default="mistral-7B-instruct", choices=sorted(OPEN_MODELS),
                   help="RAGTruth generator whose responses are evaluated")
    p.add_argument("--hf-model", default=None, help="override the Hugging Face model id")
    p.add_argument("--data-dir", default="data/ragtruth")
    p.add_argument("--split", default="test")
    p.add_argument("--tasks", default="QA,Summary,Data2txt")
    p.add_argument("--limit", type=int, default=0, help="max answers per task (0 = all)")
    p.add_argument("--labelled-only", action="store_true",
                   help="keep only responses with at least one hallucination span")
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--max-distance", type=int, default=3, help="dependency-hop threshold for edges")
    p.add_argument("--prior-weight", type=float, default=2.0, help="subjective-logic prior weight W")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--reuse-dir", default=None, help="folder searched for earlier generations.jsonl and nli.jsonl files")
    p.add_argument("--time-budget-hours", type=float, default=0,
                   help="stop generating after this many hours so later stages still run (0 = no limit)")
    p.add_argument("--nli-model", default="microsoft/deberta-large-mnli")
    p.add_argument("--nli-batch", type=int, default=16)
    p.add_argument("--nli-strength", type=float, default=10.0,
                   help="how many resamples one NLI verdict is worth when fused (default: as many as --n-samples)")
    p.add_argument("--stage", choices=["all", "generate", "nli", "evaluate"], default="all")
    p.add_argument("--out", default="runs/pilot")
    args = p.parse_args()
    args.hf_model = args.hf_model or OPEN_MODELS[args.model_label]

    per_task = [load_ragtruth(args.data_dir, args.model_label, args.split, {task}, args.limit or None,
                              args.labelled_only)
                for task in args.tasks.split(",")]  # --limit applies per task, so tasks stay balanced
    # interleave tasks, so a session cut short by the time budget still covers every task evenly
    rows = [r for group in itertools.zip_longest(*per_task) for r in group if r is not None]
    print(f"{len(rows)} responses from {args.model_label} ({args.split})")
    os.makedirs(args.out, exist_ok=True)
    gen_path = os.path.join(args.out, "generations.jsonl")
    nli_path = os.path.join(args.out, "nli.jsonl")
    if args.stage in ("all", "generate"):
        stage_generate(args, rows, gen_path)
    if args.stage in ("all", "nli"):
        stage_nli(args, rows, gen_path, nli_path)
    if args.stage in ("all", "evaluate"):
        stage_evaluate(args, rows, gen_path, nli_path)


if __name__ == "__main__":
    main()
