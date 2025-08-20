#!/usr/bin/env python3

import requests
import json
import sys

def call_retrieval_api(query, api_url="http://localhost:30888/search", k=3, n_probe=256):
    """Call the massive-serve API with a query"""
    try:
        payload = {
            "query": query,
            "n_docs": k,
            "exact_search": False,
            "diverse_search": False,
            "nprobe": n_probe,
        }
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"API call failed: {e}")
        return None

def load_offline_results(query, filepath):
    """Find the offline results for a specific query"""
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                offline_query = data['raw_query'].strip()
                if offline_query == query.strip():
                    print(f"Found matching query at line {line_num}")
                    return data
                # Check first 100 chars for debugging
                if line_num <= 3:
                    print(f"Line {line_num} query starts with: {offline_query[:100]}...")
        print(f"Query not found in {filepath}")
        return None
    except Exception as e:
        print(f"Error loading offline results: {e}")
        return None

def compare_results(api_result, offline_result, query_idx):
    """Compare API results with offline results"""
    print(f"\n=== QUERY {query_idx + 1} COMPARISON ===")
    
    if not api_result:
        print("❌ API call failed")
        return 0, 0
    
    if not offline_result:
        print("❌ Offline result not found")
        return 0, 0
    
    # Extract passage IDs from API results
    api_passages_raw = api_result.get('results', {}).get('passages', [])
    # Handle case where passages is a list containing a single list
    if len(api_passages_raw) == 1 and isinstance(api_passages_raw[0], list):
        api_passages = api_passages_raw[0]
    else:
        api_passages = api_passages_raw
    
    # Extract IDs, handling different field names
    api_ids = []
    for p in api_passages:
        if 'index_id' in p:
            api_ids.append(p['index_id'])
        elif 'id' in p:
            api_ids.append(p['id'])
        else:
            print(f"Warning: No ID field found in passage: {p.keys()}")
    
    # Extract passage IDs from offline results  
    offline_passages = offline_result.get('ctxs', [])
    offline_ids = [p.get('id') for p in offline_passages if 'id' in p]
    
    print(f"API returned {len(api_ids)} passages")
    print(f"Offline has {len(offline_ids)} passages")
    
    # Compare top 3 passage IDs
    top_k = 3
    api_top_k = api_ids[:top_k]
    offline_top_k = offline_ids[:top_k]
    
    print(f"\nTop {top_k} API passage IDs: {api_top_k}")
    print(f"Top {top_k} offline passage IDs: {offline_top_k}")
    
    # Calculate overlap
    overlap = set(api_top_k) & set(offline_top_k)
    print(f"Overlap in top {top_k}: {len(overlap)}/{top_k} passages")
    print(f"Overlapping IDs: {list(overlap)}")
    
    # Check exact order match
    if api_top_k == offline_top_k:
        print("✅ Exact order match!")
    else:
        print("❌ Different order or passages")
    
    return len(overlap), top_k

def main():
    offline_ann_file = "/home/ubuntu/CompactDS-102GB-retrieval-results/agi_eval_ann_results.jsonl"
    
    # Load first 5 offline results directly
    offline_results = []
    with open(offline_ann_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:  # Only take first 5
                break
            offline_results.append(json.loads(line.strip()))
    
    total_overlap = 0
    total_passages = 0
    
    for i, offline_result in enumerate(offline_results):
        query = offline_result['raw_query']
        print(f"\nProcessing query {i+1}/{len(offline_results)}")
        print(f"Query: {query[:100]}...")
        
        # Call API
        api_result = call_retrieval_api(query)
        
        # Compare (we already have the offline result)
        overlap, top_k = compare_results(api_result, offline_result, i)
        total_overlap += overlap
        total_passages += top_k
    
    print(f"\n=== OVERALL SUMMARY ===")
    print(f"Total overlap: {total_overlap}/{total_passages} passages")
    if total_passages > 0:
        print(f"Overall accuracy: {total_overlap/total_passages*100:.1f}%")
    else:
        print("No valid comparisons made")

if __name__ == "__main__":
    main()
