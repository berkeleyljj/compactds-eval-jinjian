# truncate.py
from typing import Iterable, Tuple, Optional

def _get_ctx(req) -> Tuple[str, Optional[str]]:
    # Supports both dict-like and attr-like request objects
    if isinstance(req, dict):
        c = req.get("context")
        i = req.get("inputs")
    else:
        c = getattr(req, "context", None)
        i = getattr(req, "inputs", None)
    if isinstance(c, str) and c:
        return c, "context"
    if isinstance(i, str) and i:
        return i, "inputs"
    return "", None

def _set_ctx(req, field: Optional[str], text: str) -> None:
    if not field:
        return
    if isinstance(req, dict):
        req[field] = text
    else:
        setattr(req, field, text)

def _ensure_tokenizer(eval_model=None, model_config=None):
    if eval_model is not None and getattr(eval_model, "tokenizer", None) is not None:
        return eval_model.tokenizer
    if model_config is None:
        return None
    try:
        from transformers import AutoTokenizer
        tk_pretrained = model_config.get("model_path") or model_config.get("model")
        return AutoTokenizer.from_pretrained(
            tk_pretrained,
            use_fast=True,
            trust_remote_code=model_config.get("trust_remote_code", False),
        )
    except Exception:
        return None

def truncate_long_contexts(
    instances: Iterable,
    *,
    target_cap: int = 13_000,
    eval_model=None,
    model_config=None,
    verbose: bool = True,
) -> int:
    """
    Truncate overly long request contexts in-place to `target_cap` tokens.
    Returns the number of contexts truncated.
    """
    tok = _ensure_tokenizer(eval_model, model_config)
    if tok is None:
        if verbose:
            print("[CONTEXT TRUNCATION] No tokenizer available; skipping.", flush=True)
        return 0

    if verbose:
        print(f"[CONTEXT TRUNCATION] Truncating contexts > {target_cap} tokens...", flush=True)

    truncated = 0
    for ins in (instances or []):
        req = getattr(ins, "request", ins)  # handle RequestInstance or raw
        text, which = _get_ctx(req)
        if not which:
            continue
        ids = tok(text, add_special_tokens=False).input_ids
        if len(ids) > target_cap:
            new_text = tok.decode(
                ids[:target_cap],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            _set_ctx(req, which, new_text)
            truncated += 1
            if verbose:
                did = getattr(ins, "doc_id", getattr(ins, "id", "?"))
                print(f"[TRUNCATE] doc_id={did}: {len(ids)} -> {target_cap}", flush=True)

    if verbose:
        if truncated:
            print(f"[CONTEXT TRUNCATION] Truncated {truncated} contexts.", flush=True)
        else:
            print("[CONTEXT TRUNCATION] No contexts needed truncation.", flush=True)
    return truncated
