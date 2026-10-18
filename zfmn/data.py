"""RAGTruth loading. RAGTruth (Niu et al. 2024) has word-level human hallucination spans."""

import json
import os
import urllib.request

RAGTRUTH_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/{name}"
FILES = ("response.jsonl", "source_info.jsonl")

# RAGTruth model label -> Hugging Face model id. Only open-weight generators are usable,
# because we score each response with the model that wrote it.
OPEN_MODELS = {
    "mistral-7B-instruct": "mistralai/Mistral-7B-Instruct-v0.1",
    "llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
}


def download_ragtruth(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    for name in FILES:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            print(f"downloading {name}")
            urllib.request.urlretrieve(RAGTRUTH_URL.format(name=name), path)


def source_text(src):
    """The grounding material the model was given, as plain text."""
    info = src["source_info"]
    if isinstance(info, str):
        return info
    if src["task_type"] == "QA":
        return info.get("passages", "")
    return json.dumps(info, ensure_ascii=False)


def load_ragtruth(data_dir, model_label, split="test", tasks=None, limit=None, labelled_only=False):
    """Return a list of dicts: id, prompt, response, spans [(start, end)], task_type."""
    download_ragtruth(data_dir)
    sources = {}
    with open(os.path.join(data_dir, "source_info.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            sources[d["source_id"]] = d
    rows = []
    with open(os.path.join(data_dir, "response.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r["model"] != model_label or r["split"] != split:
                continue
            src = sources[r["source_id"]]
            if tasks and src["task_type"] not in tasks:
                continue
            if labelled_only and not r["labels"]:
                continue
            rows.append({
                "id": r["id"],
                "task_type": src["task_type"],
                "prompt": src["prompt"],
                "response": r["response"],
                "spans": [(lab["start"], lab["end"]) for lab in r["labels"]],
                "span_types": [lab["label_type"] for lab in r["labels"]],
                "source_text": source_text(src),
            })
    rows.sort(key=lambda x: int(x["id"]))
    return rows[:limit] if limit else rows
