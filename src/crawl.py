import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from collections import defaultdict

from datetime import datetime
import time
import json
import re
import os
import random

class WebCrawler:
    def __init__(self, seed_url, max_pages=1000, max_depth=3, random_seed=42):
        """
        Initialize the web crawler.
        
        args:
            seed_url: Starting URL for crawling
            max_pages: Maximum number of pages to crawl
            max_depth: Maximum depth to crawl from seed
            random_seed: Seed for random number generator (for deterministic crawls)
        """
        self.seed_url = seed_url
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.visited = set()
        self.url_to_index = {}
        self.links = defaultdict(set)  # url -> set of outgoing urls
        self.crawl_timestamp = None
        self.rng = random.Random(random_seed)  # Deterministic random number generator
        
        # Ensure the data directory exists
        self.data_dir = os.path.join(os.path.dirname(__file__), '../data')
        os.makedirs(self.data_dir, exist_ok=True)
        
    def is_valid_url(self, url, base_domain):
        """Check if URL is valid and within the same domain."""
        parsed = urlparse(url)
        # Stay within same domain
        return parsed.netloc == base_domain and parsed.scheme in ['http', 'https']
    
    def get_links(self, url, filter_prefixes=None):
        """Extract all links from a webpage."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'
        }
        try:
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the "External links" section to stop extracting links after it
            external_links_header = soup.find('h2', id='External_links')
            
            links = set()
            after_external = external_links_header.find_all_next('a') if external_links_header else []
            for anchor in soup.find_all('a', href=True):
                # Skip links that come after the "External links" section
                if anchor in after_external:
                    continue

                link = urljoin(url, anchor['href'])
                # Remove fragments
                link = link.split('#')[0]

                # Filter out unwanted paths
                if any(prefix in link for prefix in (filter_prefixes or [])):
                    continue

                links.add(link)

            return links
        except Exception as e:
            print(f"Error crawling {url}: {e}")
            return set()
    
    def _sanitize_filename(self, url):
        """Convert URL to a safe filename component."""
        # Extract page name from URL
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        
        if path:
            # Get last segment of path
            page_name = path.split('/')[-1]
            # Remove file extensions and clean up
            page_name = re.sub(r'\.(html|php|aspx?)$', '', page_name)
            # Replace special characters with underscores
            page_name = re.sub(r'[^\w\-]', '_', page_name)
            # Limit length
            page_name = page_name[:50]
        else:
            # Use domain name if no path
            page_name = parsed.netloc.replace('.', '_')
        
        return page_name
    
    def crawl(self):
        """Perform breadth-first crawl."""
        print("\n=== Starting Crawl ===")
        print(f"Seed URL: {self.seed_url}")
        print(f"Max pages: {self.max_pages}, Max depth: {self.max_depth}")
        
        self.crawl_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_domain = urlparse(self.seed_url).netloc
        queue = [(self.seed_url, 0)]  # (url, depth)
        unique_links = set()  # Track all unique links seen during the crawl
        total_links = 0
        
        while queue and len(self.visited) < self.max_pages:
            url, depth = queue.pop(0)
            
            if url in self.visited or depth > self.max_depth:
                continue
            
            print(f"Crawling ({len(self.visited) + 1}/{self.max_pages}): {url}")
            self.visited.add(url)
            
            # Get links from this page
            filters = ["Help:", "Wikipedia:", "Talk:", "Category:", "index.php", "Special:", "File:", "Template:", "Portal:", "Template_talk:"]
            outgoing_links = self.get_links(url, filter_prefixes=filters)
            
            # Filter valid links and sort for determinism
            valid_links = [link for link in outgoing_links 
                          if self.is_valid_url(link, base_domain)]
            valid_links.sort() 
            self.rng.shuffle(valid_links)
            
            # Add all valid links to the unique_links set
            unique_links.update(valid_links)
            total_links += len(valid_links)
            
            # Store all valid links (no random selection)
            self.links[url].update(valid_links)
            print(f"  → Found {len(valid_links)} valid links")
            
            # Add valid links to the queue if not already visited
            for link in valid_links:
                if link not in self.visited:
                    queue.append((link, depth + 1))
            
            # Be polite - add delay
            time.sleep(0.5)
        
        print(f"\nCrawl complete. Visited {len(self.visited)} pages.")
        print(f"Total unique links seen: {len(unique_links)}")
        print(f"Total valid links found: {total_links}")
    
    
    def save_results_json(self):
        """Save crawl results to JSON with timestamped filename."""
        print("\n=== Saving Results to JSON ===")
        if self.crawl_timestamp is None:
            self.crawl_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create filename
        page_name = self._sanitize_filename(self.seed_url)
        filename = f"{self.crawl_timestamp}_{page_name}.json"
        filepath = os.path.join(self.data_dir, filename)
        print(f"Filename: {filepath}")

        # Prepare data
        results = {
            "metadata": {
                "seed_url": self.seed_url,
                "crawl_timestamp": self.crawl_timestamp,
                "total_pages": len(self.visited),
                "max_pages": self.max_pages,
                "max_depth": self.max_depth,
            },
            "pages": []
        }

        # Add page data
        for url in sorted(self.visited):
            page_data = {
                "url": url,
                "outgoing_links": sorted(list(self.links.get(url, set()))),
                "num_outgoing_links": len(self.links.get(url, set()))
            }

            results["pages"].append(page_data)

        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {filepath}")
        return filepath
    
    def save_data(self, filename='crawl_data.json'):
        """Save crawl data to JSON file."""
        filepath = os.path.join(self.data_dir, filename)
        data = {
            'visited': list(self.visited),
            'links': {url: list(links) for url, links in self.links.items()},
            'url_to_index': self.url_to_index
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Data saved to {filepath}")
    
    def load_data(self, filename='crawl_data.json'):
        """Load crawl data from JSON file."""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.visited = set(data['visited'])
        self.links = defaultdict(set, {url: set(links) for url, links in data['links'].items()})
        self.url_to_index = data['url_to_index']
        print(f"Data loaded from {filename}")


def main():
    # Example: Crawl Wikipedia pages starting from a topic
    seed = "https://en.wikipedia.org/wiki/Umamusume:_Pretty_Derby"
    
    crawler = WebCrawler(seed, max_pages=10, max_depth=4)
    crawler.crawl()
    
    # Save for later use
    print("\n=== Saving Crawl Data ===")
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    crawler.save_data(f"{now}_crawl_data.json")
    crawler.save_results_json()


if __name__ == "__main__":
    main()