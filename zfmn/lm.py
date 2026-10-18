"""Token-level uncertainty from a causal LM: teacher-forced scoring and sampling."""

import torch


def load_model(name, load_in_4bit=False, device=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if not tok.is_fast:
        raise ValueError("A fast tokenizer is required for character offsets.")
    kwargs = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if device and not load_in_4bit:
        model.to(device)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def format_prompt(tok, prompt):
    """Wrap the user prompt in the model's chat template when it has one."""
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    return prompt + "\n\n"


@torch.no_grad()
def score_response(model, tok, prompt, response, max_length=4096):
    """Teacher-force `response` after `prompt`.

    Returns one dict per response token: char offsets (start, end) into `response`,
    log-probability of the token, and entropy of the predictive distribution.
    """
    prompt_ids = tok(format_prompt(tok, prompt), add_special_tokens=False)["input_ids"]
    enc = tok(response, add_special_tokens=False, return_offsets_mapping=True)
    resp_ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if not resp_ids:
        return []
    keep = max_length - len(resp_ids)
    if keep < 1:
        raise ValueError("Response longer than max_length.")
    prompt_ids = prompt_ids[-keep:]  # truncate the prompt from the left if needed
    ids = torch.tensor([prompt_ids + resp_ids], device=model.device)
    logits = model(ids).logits[0].float()
    # position i predicts token i+1; response tokens start at len(prompt_ids)
    pred = logits[len(prompt_ids) - 1 : len(prompt_ids) - 1 + len(resp_ids)]
    logp = torch.log_softmax(pred, dim=-1)
    target = torch.tensor(resp_ids, device=logp.device)
    tok_lp = logp.gather(1, target[:, None])[:, 0]
    entropy = -(logp.exp() * logp).sum(-1)
    return [
        {"start": int(s), "end": int(e), "logprob": float(lp), "entropy": float(h)}
        for (s, e), lp, h in zip(offsets, tok_lp.tolist(), entropy.tolist())
    ]


@torch.no_grad()
def sample_responses(model, tok, prompt, n, temperature=0.7, max_new_tokens=300, batch_size=5):
    """Draw n sampled responses to `prompt` (texts only; score them with score_response)."""
    text = format_prompt(tok, prompt)
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    out = []
    while len(out) < n:
        k = min(batch_size, n - len(out))
        gen = model.generate(
            ids,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            num_return_sequences=k,
            pad_token_id=tok.pad_token_id,
        )
        for g in gen:
            out.append(tok.decode(g[ids.shape[1]:], skip_special_tokens=True).strip())
    return out
