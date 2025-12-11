import pagerank, crawl
import os
import json
import numpy as np

def main(seed_url="https://en.wikipedia.org/wiki/Umamusume:_Pretty_Derby"):
    
    # Example: Crawl Wikipedia pages starting from a topic
    # crawler = crawl.WebCrawler(seed_url, max_pages=1000, max_depth=3)
    # crawler.crawl()
    # path = crawler.save_results_json()

    # Load crawl data to calculate statistics
    crawl_path = "../data/20251206_192002_Umamusume__Pretty_Derby.json"
    
    with open(crawl_path, 'r') as f:
        data = json.load(f)
    
    # Calculate unique links seen statistic
    visited_urls = {page['url'] for page in data['pages']}
    all_unique_links = set()
    total_links_count = 0
    
    for page in data['pages']:
        all_unique_links.update(page['outgoing_links'])
        total_links_count += len(page['outgoing_links'])
    
    print(f"=== Crawl Statistics ===")
    print(f"Pages visited: {len(visited_urls)}")
    print(f"Total unique links seen: {len(all_unique_links)}")
    print(f"Total links count: {total_links_count}")
    
    # Build adjacency matrix from crawl results
    A, url_to_index, index_to_url = pagerank.build_adjacency_matrix(crawl_path)
    
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