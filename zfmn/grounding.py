"""Sentence-level natural-language inference (NLI) evidence for Z-FMN.

Two uses of one NLI model:

1. Source grounding. Each answer sentence is checked against the source the model was given
   (SummaC-ZS style: the source is split into chunks, and the chunk with the highest
   entailment-minus-contradiction score decides). Entailment is belief, contradiction is
   disbelief, neutral is missing evidence. This is the external evidence that model-internal
   confidence lacks, and the one that can catch confidently held contradictions.
2. SelfCheckGPT-NLI baseline (Manakul et al., 2023). Each answer sentence is checked against
   every resample; the score is the mean of P(contradiction) / (P(contradiction) + P(entailment)).

Sentence scores are mapped onto the TFMN elements whose words lie in that sentence. When an
element occurs in several sentences, its most contradicted occurrence counts.
"""

import json

from .zops import Opinion, from_evidence, fuse


def flatten_source(text):
    """Turn a JSON source (data-to-text) into 'key: value.' lines that an NLI model can read."""
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return text
    lines = []

    def walk(o, prefix=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{prefix}{k} ")
        elif isinstance(o, list):
            for v in o:
                walk(v, prefix)
        elif o not in (None, ""):
            lines.append(f"{prefix.strip()}: {o}.")

    walk(obj)
    return "\n".join(lines) if lines else text


def sentences(text, nlp):
    """Character spans of the sentences in text."""
    return [(s.start_char, s.end_char) for s in nlp(text).sents if s.text.strip()]


def chunk_text(text, nlp, max_words=300):
    """Consecutive sentence windows of at most max_words words, overlapping by one sentence."""
    sents = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    if not sents:
        return [text]
    chunks, cur, n = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and n + w > max_words:
            chunks.append(" ".join(cur))
            cur, n = cur[-1:], len(cur[-1].split())
        cur.append(s)
        n += w
    chunks.append(" ".join(cur))
    return chunks


class NLI:
    """Batched NLI with any Hugging Face MNLI classifier. Returns (entail, neutral, contra) per pair."""

    def __init__(self, name="microsoft/deberta-large-mnli", device=None, batch_size=16, max_length=512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name)
        # float32: DeBERTa's attention fails in half precision ("expected scalar type Float but found Half")
        self.model = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.float32).to(self.device).eval()
        labels = {i: l.lower() for i, l in self.model.config.id2label.items()}
        self.idx = {k: next(i for i, l in labels.items() if l.startswith(k)) for k in ("entail", "neutral", "contra")}
        self.batch_size, self.max_length = batch_size, max_length

    def __call__(self, pairs):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                prem, hyp = zip(*pairs[i:i + self.batch_size])
                # truncate the premise only, so the hypothesis is always read in full
                enc = self.tok(list(prem), list(hyp), truncation="only_first", max_length=self.max_length,
                               padding=True, return_tensors="pt").to(self.device)
                p = self.model(**enc).logits.float().softmax(-1).tolist()
                out += [(r[self.idx["entail"]], r[self.idx["neutral"]], r[self.idx["contra"]]) for r in p]
        return out


def nli_record(response, source, samples, nlp, nli):
    """Everything NLI says about one answer: per-sentence source grounding and SelfCheckGPT scores."""
    sents = sentences(response, nlp)
    hyps = [response[s:e] for s, e in sents]
    chunks = chunk_text(flatten_source(source), nlp)
    sample_texts = [s for s in samples if s.strip()]
    pairs = [(c, h) for h in hyps for c in chunks] + [(t, h) for h in hyps for t in sample_texts]
    probs = nli(pairs) if pairs else []
    nc, ns = len(chunks), len(sample_texts)
    src, selfcheck = [], []
    for i in range(len(hyps)):
        per_chunk = probs[i * nc:(i + 1) * nc]
        e, n, c = max(per_chunk, key=lambda p: p[0] - p[2])
        src.append([e, n, c, max(p[2] for p in per_chunk)])
        off = len(hyps) * nc + i * ns
        per_sample = probs[off:off + ns]
        selfcheck.append(sum(c / (c + e + 1e-9) for e, _, c in per_sample) / ns if ns else None)
    return {"sents": sents, "src": src, "selfcheck": selfcheck}


def _sentence_of(span, sents):
    for i, (s, e) in enumerate(sents):
        if span[0] < e and span[1] > s:
            return i
    return None


def element_nli(occ_spans, rec):
    """NLI fields for one element, given the char spans of its occurrences in the answer."""
    idx = {_sentence_of(sp, rec["sents"]) for sp in occ_spans} - {None}
    if not idx:
        return None
    i = max(idx, key=lambda j: rec["src"][j][2])  # most contradicted occurrence
    e, n, c, cmax = rec["src"][i]
    sc = [rec["selfcheck"][j] for j in idx if rec["selfcheck"][j] is not None]
    return {"nli_e": e, "nli_n": n, "nli_c": c, "nli_cmax": cmax,
            "selfcheck": max(sc) if sc else None}


def nli_opinion(e, c, strength=10.0, W=2.0, a=0.5):
    """NLI probabilities as subjective-logic evidence worth `strength` observations.

    Entailment is positive evidence, contradiction negative, and neutral is no evidence, so a
    neutral verdict leaves the opinion uncertain instead of pushing it either way.
    """
    return from_evidence(strength * e, strength * c, W=W, a=a)


def fused_fields(sl_b, sl_u, e, c, strength=10.0, W=2.0, a=0.5, selfcheck=None, n_samples=10):
    """Model opinion fused with source entailment, and optionally with sample consistency.

    The three-way fusion adds SelfCheckGPT's sentence-level verdict as a third opinion worth
    n_samples observations: consistent resamples are belief, contradicting ones disbelief. It is
    semantic agreement with the samples, which the word-level model opinion does not measure.
    """
    model = Opinion(sl_b, max(0.0, 1 - sl_b - sl_u), sl_u, a)
    f = fuse(model, nli_opinion(e, c, strength, W, a))
    out = {"nli_fused_P": f.projected, "nli_fused_d": f.d, "nli_fused_u": f.u}
    if selfcheck is not None:
        f3 = fuse(f, from_evidence(n_samples * (1 - selfcheck), n_samples * selfcheck, W=W, a=a))
        out.update({"tri_fused_P": f3.projected, "tri_fused_d": f3.d, "tri_fused_u": f3.u})
    return out
