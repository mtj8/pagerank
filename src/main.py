import pagerank, crawl
import os
import json
import numpy as np

def main(seed_url="https://en.wikipedia.org/wiki/Umamusume:_Pretty_Derby"):
    # Example: Crawl Wikipedia pages starting from a topic
    # crawler = crawl.WebCrawler(seed_url, max_pages=1000, max_depth=3)
    # crawler.crawl()
    # path = crawler.save_results_json()
    
    # Build adjacency matrix from crawl results
    A, url_to_index, index_to_url = pagerank.build_adjacency_matrix("../data/20251201_163213_Umamusume__Pretty_Derby.json")
    
    # Compute PageRank
    R: np.ndarray = pagerank.page_rank(A, eps=1e-9, max_iters=100)
    
    # Save results
    ranked_indices = R.flatten().argsort()[::-1]
    results = {}
    for rank, idx in enumerate(ranked_indices, start=1):
        url = index_to_url[idx]
        score = R[idx][0]
        results[rank] = (url, score)
    
    with open(f"../data/pagerank_results_{seed_url.split('/')[-1]}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"PageRank results saved to pagerank_results_{seed_url.split('/')[-1]}.json")

if __name__ == "__main__":
    main()