# modules/retriever/massive_serve.py

import requests
import json

class MassiveServeRetriever:
    def __init__(self, api_url, k=10, use_rerank=False, n_probe=None):
        self.api_url = api_url
        self.k = k
        self.use_rerank = use_rerank
        self.n_probe = n_probe

    def retrieve(self, query):
        print(f"At API calling n_probe is set to {self.n_probe}")
        response = requests.post(self.api_url, json={"query": query, "n_docs": self.k, "use_rerank": self.use_rerank, "nprobe": self.n_probe})
        response.raise_for_status()

        result = response.json()

        # ✅ debug print
        # print(f"[MassiveServeRetriever] Retrieved:\n{json.dumps(result, indent=2)[:1000]}")

        # ✅ MUST return a Python dict like {"scores": ..., "passages": ...}
        return result