import requests
import json

class MassiveServeRetriever:
    def __init__(self, api_url, k=10, use_rerank=False, use_diverse=False, n_probe=None, retrieval_batch_size=1):
        self.api_url = api_url
        self.k = k
        self.use_rerank = use_rerank
        self.use_diverse = use_diverse
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
                    "exact_search": self.use_rerank,
                    "diverse_search": self.use_diverse,
                    "nprobe": self.n_probe,
                },
            )
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
                response = requests.post(
                    self.api_url,
                    json={
                        "queries": subqueries,
                        "n_docs": self.k,
                        "exact_search": self.use_rerank,
                        "diverse_search": self.use_diverse,
                        "nprobe": self.n_probe,
                    },
                )
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
