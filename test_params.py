#!/usr/bin/env python3

import argparse
import sys
import os
sys.path.insert(0, "/home/ubuntu/Jinjian/compactds-eval")

from modules.retriever.massive_serve import MassiveServeRetriever

def test_parameters():
    # Test 1: Both parameters disabled (default)
    print("=== Test 1: Both parameters disabled ===")
    retriever = MassiveServeRetriever(
        api_url="http://localhost:30888/search",
        k=3,
        use_rerank=False,
        use_diverse=False,
        n_probe=256
    )
    print(f"use_rerank: {retriever.use_rerank}")
    print(f"use_diverse: {retriever.use_diverse}")
    
    # Test 2: Both parameters enabled
    print("\n=== Test 2: Both parameters enabled ===")
    retriever = MassiveServeRetriever(
        api_url="http://localhost:30888/search",
        k=3,
        use_rerank=True,
        use_diverse=True,
        n_probe=256
    )
    print(f"use_rerank: {retriever.use_rerank}")
    print(f"use_diverse: {retriever.use_diverse}")
    
    # Test 3: Only exact_search enabled
    print("\n=== Test 3: Only exact_search enabled ===")
    retriever = MassiveServeRetriever(
        api_url="http://localhost:30888/search",
        k=3,
        use_rerank=True,
        use_diverse=False,
        n_probe=256
    )
    print(f"use_rerank: {retriever.use_rerank}")
    print(f"use_diverse: {retriever.use_diverse}")

if __name__ == "__main__":
    test_parameters()
