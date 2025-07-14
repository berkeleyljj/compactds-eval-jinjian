import requests
import json

class MassiveServeRetriever:
    def __init__(self, api_url, k=10, use_rerank=False, n_probe=None, retrieval_batch_size=1):
        self.api_url = api_url
        self.k = k
        self.use_rerank = use_rerank
        self.n_probe = n_probe
        self.retrieval_batch_size = retrieval_batch_size
        self.batched_results = None  # to be populated if batch retrieval is used

    def retrieve(self, query):
        if self.retrieval_batch_size == 1:
            print("[Retriever] Using per-query retrieval")
            response = requests.post(
                self.api_url,
                json={
                    "query": query,
                    "n_docs": self.k,
                    "use_rerank": self.use_rerank,
                    "nprobe": self.n_probe,
                },
            )
            response.raise_for_status()
            return response.json()
        else:
            raise ValueError("Per-query retrieve() shouldn't be called when retrieval_batch_size > 1")

    def retrieve_batch(self, queries):
        print(f"[Retriever] Using batched retrieval with batch size = {self.retrieval_batch_size} to api url {self.api_url}")
        response = requests.post(
            self.api_url,
            json={
                "queries": queries,
                "n_docs": self.k,
                "use_rerank": self.use_rerank,
                "nprobe": self.n_probe,
            },
        )
        response.raise_for_status()
        json_response = response.json()

        results = json_response.get("results", {})
        # print(results)
        passages = results.get("passages", [])
        print(f"[DEBUG] Received {len(passages)} passage groups (should match query batch size)")

        return json_response
