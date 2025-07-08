# modules/retriever/massive_serve.py

import requests

class MassiveServeRetriever:
    def __init__(self, api_url: str, k: int):
        self.api_url = api_url
        self.k = k

    def retrieve(self, query: str):
        resp = requests.post(
            self.api_url,
            json={"query": query, "k": self.k},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()["passages"]  # assumes server returns JSON with "passages" list
