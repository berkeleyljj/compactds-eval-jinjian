#!/usr/bin/env python3

import requests
import json

def analyze_retrieval_differences():
    """Detailed analysis of retrieval differences between API and offline results"""
    
    # Load first query that had a mismatch (Query 2)
    offline_file = "/home/ubuntu/CompactDS-102GB-retrieval-results/agi_eval_ann_results.jsonl"
    
    with open(offline_file, 'r') as f:
        # Get the second query (index 1)
        for i, line in enumerate(f):
            if i == 1:  # Second query that showed partial match
                offline_result = json.loads(line.strip())
                break
    
    query = offline_result['raw_query']
    print(f"Analyzing query: {query[:100]}...")
    
    # Call API with exact same parameters as evaluation
    api_response = requests.post(
        "http://localhost:30888/search",
        json={
            "query": query,
            "n_docs": 1000,  # Get more results to see full ranking
            "exact_search": False,
            "diverse_search": False,
            "nprobe": 256,
        },
        timeout=60
    )
    
    api_result = api_response.json()
    api_passages = api_result['results']['passages'][0] if api_result['results']['passages'] else []
    
    # Extract top 10 from both
    api_top_10 = [p['index_id'] for p in api_passages[:10]]
    offline_top_10 = [p['id'] for p in offline_result['ctxs'][:10]]
    
    print(f"\nAPI Top 10 IDs: {api_top_10}")
    print(f"Offline Top 10 IDs: {offline_top_10}")
    
    # Check if offline top results appear anywhere in API results
    print(f"\nOffline results position in API ranking:")
    api_all_ids = [p['index_id'] for p in api_passages]
    
    for i, offline_id in enumerate(offline_top_10):
        if offline_id in api_all_ids:
            api_position = api_all_ids.index(offline_id)
            print(f"Offline #{i+1} ({offline_id}) → API position #{api_position+1}")
        else:
            print(f"Offline #{i+1} ({offline_id}) → NOT FOUND in API results")
    
    # Check API vs offline scoring for overlapping passages
    print(f"\nScoring comparison for top 3 overlapping passages:")
    
    for i in range(min(3, len(api_passages))):
        api_passage = api_passages[i]
        api_id = api_passage['index_id']
        
        # Find corresponding offline passage
        offline_passage = None
        for offline_p in offline_result['ctxs']:
            if offline_p['id'] == api_id:
                offline_passage = offline_p
                break
        
        if offline_passage:
            print(f"\nPassage ID {api_id}:")
            print(f"  API position: #{i+1}")
            offline_pos = [p['id'] for p in offline_result['ctxs']].index(api_id) + 1
            print(f"  Offline position: #{offline_pos}")
            
            # Check if we have scoring info
            if 'retrieval score' in offline_passage:
                print(f"  Offline score: {offline_passage['retrieval score']}")
            
            # Compare passage text snippets
            api_text = api_passage.get('text', '')[:100]
            offline_text = offline_passage.get('retrieval text', '')[:100]
            print(f"  Text match: {api_text == offline_text}")

if __name__ == "__main__":
    analyze_retrieval_differences()
