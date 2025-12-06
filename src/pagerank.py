import numpy as np
import math
import json

DAMPING_FACTOR = 0.85
MAX_ITERS = 100
TOLERANCE = 1.0e-9





def page_rank(A: np.array, eps: float, max_iters: int) -> np.array:
    """
    Compute the PageRank vector R.

    args:
        A: normalized adjacency matrix (n x n)
        eps: convergence tolerance
        max_iters: maximum number of iterations
    
    returns:
        R: PageRank vector (n x 1)
    """
    n = A.shape[0]

    E = np.array([[1/n] for _ in range(n)])  # uniform "random surfer" vector

    R = E.copy()  # initial PageRank vector

    for iter in range(max_iters):
        R_new = (DAMPING_FACTOR * A @ R) + (1 - DAMPING_FACTOR) * E

        # convergence
        delta = np.linalg.norm(R_new - R, 1)
        if delta < eps:
            break

        R = R_new

    return R


def build_adjacency_matrix(crawl_json_path):
    """
    Build a normalized adjacency matrix from crawl results JSON.
    
    args:
        crawl_json_path: Path to the JSON file produced by WebCrawler.save_results_json()
    
    returns:
        A: Normalized adjacency matrix (n x n) where A[i][j] represents 
           the probability of going from page j to page i
        url_to_index: Dictionary mapping URLs to matrix indices
        index_to_url: Dictionary mapping matrix indices to URLs
    """
    # Load crawl data
    with open(crawl_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pages = data['pages']
    n = len(pages)
    
    # Build URL to index mapping - only include visited pages
    visited_urls = {page['url'] for page in pages}
    url_to_index = {page['url']: idx for idx, page in enumerate(pages)}
    index_to_url = {idx: url for url, idx in url_to_index.items()}
    
    # Initialize adjacency matrix
    A = np.zeros((n, n))
    
    # Build the matrix
    for page in pages:
        source_url = page['url']
        source_idx = url_to_index[source_url]
        
        # Get outgoing links that are in the visited set (excluding self-links)
        outgoing_links = [link for link in page['outgoing_links'] 
                         if link in visited_urls and link != source_url]
        
        out_degree = len(outgoing_links)
        
        if out_degree > 0:
            # For each outgoing link, add edge with weight 1/out_degree
            for target_url in outgoing_links:
                target_idx = url_to_index[target_url]
                A[target_idx][source_idx] = 1.0 / out_degree
        else:
            # Dangling node: distribute probability uniformly to all pages
            A[:, source_idx] = 1.0 / n
    
    return A, url_to_index, index_to_url


def build_adjacency_list(crawl_json_path, output_json_path=None):
    """
    Build an adjacency list from crawl results JSON for visited URLs only.
    
    args:
        crawl_json_path: Path to the JSON file produced by WebCrawler.save_results_json()
        output_json_path: Optional path to save the adjacency list JSON. If None, doesn't save to file.
    
    returns:
        adjacency_list: Dictionary mapping each URL to a list of URLs it links to (within visited set)
    """
    # Load crawl data
    with open(crawl_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pages = data['pages']
    
    # Build set of visited URLs
    visited_urls = {page['url'] for page in pages}
    
    # Build adjacency list - only include links to visited pages (excluding self-links)
    adjacency_list = {}
    for page in pages:
        source_url = page['url']
        # Filter outgoing links to only those in the visited set, excluding self-links
        outgoing_links = [link for link in page['outgoing_links'] 
                         if link in visited_urls and link != source_url]
        adjacency_list[source_url] = outgoing_links
    
    # Optionally save to file
    if output_json_path:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(adjacency_list, f, indent=2, ensure_ascii=False)
        print(f"Adjacency list saved to {output_json_path}")
    
    return adjacency_list


def main():
    # small test
    test1 = np.array([[0, 1/2, 1/2, 0],
                      [1/3, 0, 0, 1/3],
                      [1/3, 1/2, 0, 1/3],
                      [1/3, 0, 1/2, 1/3]])
    pr1 = page_rank(test1, TOLERANCE, MAX_ITERS)
    print("Test 1 PageRank:\n", pr1)
    print("Sanity check: sum =", np.sum(pr1))
    
    # load crawl data and compute pagerank
    crawl_file = "../data/20251201_161020_Umamusume__Pretty_Derby.json"
    A, url_to_index, index_to_url = build_adjacency_matrix(crawl_file)
    pr = page_rank(A, TOLERANCE, MAX_ITERS)
    
    # show top 10 pages by pagerank
    ranked_indices = np.argsort(pr.flatten())[::-1][:10]
    print("\nTop 10 pages by PageRank:")
    for i, idx in enumerate(ranked_indices, 1):
        print(f"{i}. {index_to_url[idx]}: {pr[idx][0]:.6f}")
    
if __name__ == "__main__":
    # main()
    build_adjacency_list("../data/20251201_163213_Umamusume__Pretty_Derby.json", "../data/adjacency_list_Umamusume__Pretty_Derby.json")