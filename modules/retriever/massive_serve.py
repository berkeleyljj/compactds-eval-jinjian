import requests
import json
import os

class MassiveServeRetriever:
    def __init__(self, api_url, k=10, use_rerank=False, use_diverse=False, n_probe=None, retrieval_batch_size=1, lambda_val=None,
                 backend=None, diskann_L=None, diskann_W=None, diskann_threads=None):
        self.api_url = api_url
        self.k = k
        self.use_rerank = use_rerank
        self.use_diverse = use_diverse
        self.n_probe = n_probe
        self.retrieval_batch_size = retrieval_batch_size
        self.lambda_val = lambda_val
        self.batched_results = None  # to be populated if batch retrieval is used

        # Backend selection and DiskANN params (allow env override)
        self.backend = backend or os.environ.get("MS_BACKEND")
        # Coerce env strings to ints when present
        def _as_int(val):
            try:
                return int(val) if val is not None else None
            except Exception:
                return None

        self.diskann_L = diskann_L if diskann_L is not None else _as_int(os.environ.get("DISKANN_L"))
        self.diskann_W = diskann_W if diskann_W is not None else _as_int(os.environ.get("DISKANN_W"))
        self.diskann_threads = diskann_threads if diskann_threads is not None else _as_int(os.environ.get("DISKANN_THREADS"))

    def _maybe_apply_backend(self, payload):
        if self.backend:
            payload["backend"] = self.backend
            if self.backend.lower() == "diskann":
                if self.diskann_L is not None:
                    payload["diskann_L"] = self.diskann_L
                if self.diskann_W is not None:
                    payload["diskann_W"] = self.diskann_W
                if self.diskann_threads is not None:
                    payload["diskann_threads"] = self.diskann_threads
        return payload

    def retrieve(self, query):
        if self.retrieval_batch_size == 1:
            print("[Retriever] Using per-query retrieval")
            payload = {
                "query": query,
                "n_docs": self.k,
                "exact_search": self.use_rerank,
                "diverse_search": self.use_diverse,
                "nprobe": self.n_probe,
            }
            if self.lambda_val is not None:
                payload["lambda"] = self.lambda_val
            payload = self._maybe_apply_backend(payload)
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json()
        else:
            raise ValueError("Per-query retrieve() shouldn't be called when retrieval_batch_size > 1")

    def retrieve_batch(self, queries):
        print(f"[Retriever] Using batched retrieval with batch size = {self.retrieval_batch_size} to api url {self.api_url}")
        all_passages = []

        for i in range(0, len(queries), self.retrieval_batch_size):
            subqueries = queries[i: i + self.retrieval_batch_size]
            try:
                payload = {
                    "queries": subqueries,
                    "n_docs": self.k,
                    "exact_search": self.use_rerank,
                    "diverse_search": self.use_diverse,
                    "nprobe": self.n_probe,
                }
                if self.lambda_val is not None:
                    payload["lambda"] = self.lambda_val
                payload = self._maybe_apply_backend(payload)
                response = requests.post(self.api_url, json=payload)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print("[ERROR] Backend request failed")
                print(e)
                raise

            json_response = response.json()
            results = json_response.get("results", {})
            batch_passages = results.get("passages", [])
            print(f"[DEBUG] Retrieved {len(batch_passages)} passages for batch of size {len(subqueries)}")
            all_passages.extend(batch_passages)

        return {
            "results": {
                "passages": all_passages
            }
        }
