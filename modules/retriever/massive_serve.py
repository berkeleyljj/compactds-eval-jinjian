# modules/retriever/massive_serve.py

import requests
import json

class MassiveServeRetriever:
    def __init__(self, api_url, k=10):
        self.api_url = api_url
        self.k = k

    def retrieve(self, query):
        response = requests.post(self.api_url, json={"query": query, "n_docs": self.k})
        response.raise_for_status()

        result = response.json()

        # ✅ debug print
        # print(f"[MassiveServeRetriever] Retrieved:\n{json.dumps(result, indent=2)[:1000]}")

        # ✅ MUST return a Python dict like {"scores": ..., "passages": ...}
        return result