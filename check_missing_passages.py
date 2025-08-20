#!/usr/bin/env python3

import requests
import json

def check_missing_passages():
    """Check if missing passages exist in the API index"""
    
    # Missing passage IDs from our analysis
    missing_ids = [853126295, 832776517, 809120027, 780069180]
    
    # Load the offline result to get the text of missing passages
    offline_file = "/home/ubuntu/CompactDS-102GB-retrieval-results/agi_eval_ann_results.jsonl"
    
    with open(offline_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 1:  # Second query
                offline_result = json.loads(line.strip())
                break
    
    print("Checking missing passages by searching for their text...")
    
    for missing_id in missing_ids:
        # Find the missing passage in offline results
        missing_passage = None
        for p in offline_result['ctxs']:
            if p['id'] == missing_id:
                missing_passage = p
                break
        
        if missing_passage:
            # Extract a unique snippet from the passage text
            passage_text = missing_passage.get('retrieval text', '')
            # Use first 50 chars as search query
            search_snippet = passage_text[:50].strip()
            
            print(f"\nSearching for passage ID {missing_id}")
            print(f"Search snippet: '{search_snippet}'")
            
            try:
                response = requests.post(
                    "http://localhost:30888/search",
                    json={
                        "query": search_snippet,
                        "n_docs": 20,
                        "exact_search": False,
                        "diverse_search": False,
                        "nprobe": 256,
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    passages = result['results']['passages'][0] if result['results']['passages'] else []
                    
                    # Check if the missing ID appears in results
                    found_ids = [p['index_id'] for p in passages]
                    if missing_id in found_ids:
                        position = found_ids.index(missing_id) + 1
                        print(f"✅ Found at position #{position}")
                    else:
                        print(f"❌ NOT FOUND in top 20 results")
                        print(f"   Top 5 IDs: {found_ids[:5]}")
                else:
                    print(f"❌ API error: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_missing_passages()
