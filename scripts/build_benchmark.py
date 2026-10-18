"""Build the RAGTruth-FMN release: concept- and link-level hallucination labels plus baseline scores.

Usage:
  python build_benchmark.py OUT_DIR Llama-2-7B=runs/full_llama-2-7b-chat Mistral-7B=runs/full_mistral-7B-instruct ...

Writes OUT_DIR/ragtruth_fmn_elements.csv.gz (one row per node or edge of each answer's textual
forma mentis network), OUT_DIR/ragtruth_fmn_answers.csv.gz (one row per answer) and
OUT_DIR/stats.md. Labels are derived from RAGTruth's human span annotations: an element is
hallucinated when one of its occurrences overlaps an annotated span.
"""
import os
import sys

import pandas as pd

EL = {"kind": "element_type", "key": "element", "n_occ": "occurrences", "wrong": "hallucinated",
      "types": "hallucination_types", "degree": "degree", "mean_lp": "score_mean_logprob",
      "selfcheck": "score_selfcheck_nli", "nli_e": "score_nli_entail", "nli_c": "score_nli_contra",
      "sl_P": "score_zfmn_model_P", "ctx_P": "score_zfmn_context_P", "ctx_u": "score_zfmn_context_u",
      "tri_fused_P": "score_zfmn_full_P", "tri_fused_d": "score_zfmn_full_d", "tri_fused_u": "score_zfmn_full_u",
      "nli_n": "score_nli_neutral", "src_P": "score_zfmn_source_P", "src_u": "score_zfmn_source_u",
      "valence": "emo_valence", "n_emotions": "emo_n_emotions", "frame_valence": "emo_frame_valence",
      "frame_valence_spread": "emo_frame_valence_spread", "response_id": "response_id", "task_type": "task"}

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
els, ans = [], []
for arg in sys.argv[2:]:
    model, folder = arg.split("=", 1)
    e = pd.read_csv(f"{folder}/elements.csv", usecols=lambda c: c in EL).rename(columns=EL)
    e.insert(0, "model", model)
    els.append(e)
    a = pd.read_csv(f"{folder}/responses.csv")
    a.insert(0, "model", model)
    ans.append(a.rename(columns={"wrong": "hallucinated", "task_type": "task"}))
E = pd.concat(els, ignore_index=True)
A = pd.concat(ans, ignore_index=True)
E["hallucination_types"] = E["hallucination_types"].fillna("")
E.to_csv(f"{out}/ragtruth_fmn_elements.csv.gz", index=False, compression="gzip")
A.to_csv(f"{out}/ragtruth_fmn_answers.csv.gz", index=False, compression="gzip")

g = E.groupby(["model", "task", "element_type"]).agg(elements=("hallucinated", "size"),
                                                     hallucinated=("hallucinated", "sum")).reset_index()
g["rate"] = (g.hallucinated / g.elements).round(3)
na = A.groupby(["model", "task"]).agg(answers=("hallucinated", "size"), with_hallucination=("hallucinated", "sum"))
with open(f"{out}/stats.md", "w") as f:
    f.write(f"# RAGTruth-FMN statistics\n\n{len(A)} answers, {len(E)} elements "
            f"({(E.element_type == 'node').sum()} concepts, {(E.element_type == 'edge').sum()} links).\n\n")
    f.write("## Answers\n\n" + na.to_markdown() + "\n\n## Elements\n\n" + g.to_markdown(index=False) + "\n")
print(open(f"{out}/stats.md").read())
