# exact_rerank.py
# --------------------------------------------------------------------------------------
# CHANGES SUMMARY
# [MODIFIED] load_embedding_from_cache: uses np.load(..., mmap_mode='r') to keep RAM flat
# [MODIFIED] save_embedding_to_cache: stores float32 to halve memory
# [MODIFIED] embed_passages: adds clean cache hit/miss prints and saved count
# [ADDED]    _batched_cosine_similarity: constant-memory cosine with query vs contexts
# [MODIFIED] exact_rerank_topk: uses batched cosine (no giant vstack / sklearn buffers),
#            casts to float32, and performs explicit cleanup (gc + torch.cuda.empty_cache)
# --------------------------------------------------------------------------------------

import os
import json
import pickle as pkl
import logging
from tqdm import tqdm
import re
import gc  # [ADDED] for explicit cleanup

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # (no longer used; safe to keep)
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from rerank import contriever
from normalize_text import normalize as nm

os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

CACHE_DIR = "./embedding_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# [MODIFIED]: Use memmap to avoid loading entire arrays into RAM permanently.
def load_embedding_from_cache(passage_id, mmap=True):
    cache_path = os.path.join(CACHE_DIR, f"{passage_id}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path, mmap_mode='r') if mmap else np.load(cache_path)
    return None

# [MODIFIED]: Force float32 to reduce memory footprint and keep dtype consistent.
def save_embedding_to_cache(passage_id, embedding):
    cache_path = os.path.join(CACHE_DIR, f"{passage_id}.npy")
    np.save(cache_path, np.asarray(embedding, dtype=np.float32))

# [MODIFIED]: Clean cache-usage printing and consistent dtype handling.
def embed_passages(args, raw_query, texts, passage_ids, model):
    # Decide what we can pull from cache vs. what we need to compute
    texts_to_embed = []
    ids_to_embed = []
    embeddings = {}

    # Always (re)compute the query; we do not cache queries
    texts_to_embed.append(raw_query)
    ids_to_embed.append(None)

    cached_hit_ids = []
    cached_miss_ids = []

    for text, pid in zip(texts, passage_ids):
        cached = load_embedding_from_cache(pid)  # memmap read
        if cached is not None:
            embeddings[text] = cached
            cached_hit_ids.append(pid)
        else:
            texts_to_embed.append(text)
            ids_to_embed.append(pid)
            cached_miss_ids.append(pid)

    total = len(texts)
    hits = len(cached_hit_ids)
    misses = len(cached_miss_ids)

    # Clean, single-line summary of cache usage
    print(f"[CACHE] Using cached embeddings for {hits}/{total} passages; "
          f"computing {misses} new passage embeddings + 1 query.")
    if hits > 0:
        print(f"[CACHE] Cache directory: {os.path.abspath(CACHE_DIR)}")

    # Embed query + cache-miss passages
    if texts_to_embed:
        encoded = model.encode(
            texts_to_embed,
            batch_size=min(128, getattr(args, "per_gpu_batch_size", 128)),
            instruction="<|embed|>\n"
        )
        saved_count = 0
        for pid, emb, original_text in zip(ids_to_embed, encoded, texts_to_embed):
            if np.isnan(emb).any():
                logging.warning(f"[WARNING] Skipping text due to NaN in embedding: {original_text}")
                continue
            embeddings[original_text] = np.asarray(emb, dtype=np.float32)
            # Cache only passage embeddings (pid != None)
            if pid is not None:
                save_embedding_to_cache(pid, emb)
                saved_count += 1

        if saved_count > 0:
            print(f"[CACHE] Saved {saved_count} new passage embeddings to cache.")

    return embeddings

# [ADDED]: Constant-memory cosine (query vs. contexts) with batching.
def _batched_cosine_similarity(query_vec, ctx_texts, text_to_embeddings, batch_size=2048):
    """
    query_vec: (1, D) float32
    ctx_texts: list[str] of normalized context texts (keys into text_to_embeddings)
    text_to_embeddings: dict[text] -> np.ndarray or np.memmap (D,)
    """
    n = len(ctx_texts)
    scores = np.empty(n, dtype=np.float32)

    # Normalize query once
    q = query_vec.astype(np.float32, copy=False)
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(q_norm, 1e-12)  # (1, D)

    for i in range(0, n, batch_size):
        chunk_texts = ctx_texts[i:i + batch_size]

        # Materialize only the needed chunk into RAM and cast to float32
        # Shape: (B, D)
        chunk = np.vstack([
            np.asarray(text_to_embeddings[t], dtype=np.float32)
            for t in chunk_texts
        ])

        # L2-normalize rows for cosine
        denom = np.linalg.norm(chunk, axis=1, keepdims=True)
        chunk = chunk / np.maximum(denom, 1e-12)

        # Cosine with single query: (1, D) @ (D, B) -> (1, B)
        chunk_scores = (q @ chunk.T)[0]  # (B,)

        scores[i:i + len(chunk_texts)] = chunk_scores

        # Free asap
        del chunk, denom, chunk_scores

    return scores

# Defaults to memory reranking with live passages
def exact_rerank_topk(raw_passages, query_encoder):
    logging.info(f"Doing exact reranking for {len(raw_passages)} total evaluation samples...")

    raw_query = raw_passages[0]['raw_query']

    # Pre-normalize contexts (same as your original)
    ctxs = [nm(p["text"].lower()) for p in raw_passages]
    ctx_ids = [p["passage_id"] for p in raw_passages]

    from types import SimpleNamespace
    args = SimpleNamespace(
        per_gpu_batch_size=128,
        get=lambda k, default=None: default
    )

    print("[INFO] Embedding query and contexts (with cache)...")
    text_to_embeddings = embed_passages(
        args=args,
        raw_query=raw_query,
        texts=ctxs,
        passage_ids=ctx_ids,
        model=query_encoder
    )

    # Build query embedding (float32)
    if raw_query not in text_to_embeddings:
        raise RuntimeError("Query embedding not found in text_to_embeddings (unexpected).")

    query_embedding = np.asarray(text_to_embeddings[raw_query], dtype=np.float32).reshape(1, -1)

    # Sanity check: ensure all ctxs are present
    missing_any = False
    for text in ctxs:
        if text not in text_to_embeddings:
            missing_any = True
            break
    if missing_any:
        print("[WARN] Missing one or more context embeddings.")
        print(f"[INFO] Embedding cache directory: {os.path.abspath(CACHE_DIR)}")

    print("[INFO] Calculating exact cosine similarities in batches...")
    similarities = _batched_cosine_similarity(
        query_embedding, ctxs, text_to_embeddings, batch_size=2048
    )

    # Assign scores (float)
    for i, p in enumerate(raw_passages):
        s = similarities[i]
        p["exact score"] = float(s) if not np.isnan(s) else -1.0

    # Sort by exact cosine score (desc)
    raw_passages.sort(key=lambda p: p["exact score"], reverse=True)

    # [MODIFIED]: Explicit cleanup to prevent gradual RAM growth across eval loop
    del text_to_embeddings, query_embedding, similarities
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return raw_passages

def normalize_text(text):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(lower(text)))

def search_topk(cfg):
    exact_rerank_topk(cfg)
