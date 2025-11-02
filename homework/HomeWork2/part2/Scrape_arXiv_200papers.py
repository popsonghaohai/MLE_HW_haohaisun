import requests
import json
import time
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
from urllib.parse import urljoin
import trafilatura
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
import pytesseract
import io


class ArxivScraper:
    def __init__(self, output_dir="arxiv_data"):
        self.base_url = "http://export.arxiv.org/api/query"
        self.abs_base_url = "https://arxiv.org/abs/"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def fetch_papers(self, category, max_results=200):
        """
        Fetch papers from arXiv API for a given category
        Args:
            category: e.g., 'cs.CL', 'cs.AI', 'math.CO'
            max_results: number of papers to fetch (default 200)
        """
        params = {
            'search_query': f'cat:{category}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }

        print(f"Fetching {max_results} papers from {category}...")
        response = requests.get(self.base_url, params=params)

        if response.status_code != 200:
            print(f"Error fetching papers: {response.status_code}")
            return []

        return self.parse_arxiv_response(response.text)

    def parse_arxiv_response(self, xml_content):
        """Parse arXiv API XML response"""
        root = ET.fromstring(xml_content)
        namespace = {'atom': 'http://www.w3.org/2005/Atom',
                     'arxiv': 'http://arxiv.org/schemas/atom'}

        papers = []
        for entry in root.findall('atom:entry', namespace):
            paper = {}

            # Extract ID/URL
            id_element = entry.find('atom:id', namespace)
            if id_element is not None:
                paper['url'] = id_element.text
                paper['arxiv_id'] = paper['url'].split('/abs/')[-1]

            # Extract title
            title_element = entry.find('atom:title', namespace)
            if title_element is not None:
                paper['title'] = ' '.join(title_element.text.split())

            # Extract abstract
            summary_element = entry.find('atom:summary', namespace)
            if summary_element is not None:
                paper['abstract'] = ' '.join(summary_element.text.split())

            # Extract authors
            authors = []
            for author in entry.findall('atom:author', namespace):
                name_element = author.find('atom:name', namespace)
                if name_element is not None:
                    authors.append(name_element.text)
            paper['authors'] = authors

            # Extract published date
            published_element = entry.find('atom:published', namespace)
            if published_element is not None:
                paper['date'] = published_element.text.split('T')[0]

            papers.append(paper)

        print(f"Parsed {len(papers)} papers from API")
        return papers

    def scrape_abstract_page(self, paper):
        """
        Scrape the /abs/ page using Trafilatura for enhanced content extraction
        """
        abs_url = paper['url']
        print(f"Scraping: {paper['title'][:60]}...")

        try:
            response = requests.get(abs_url, timeout=10)
            if response.status_code != 200:
                print(f"  Error fetching {abs_url}: {response.status_code}")
                return paper

            # Use Trafilatura to extract clean content
            downloaded = trafilatura.fetch_url(abs_url)
            if downloaded:
                extracted = trafilatura.extract(downloaded, include_comments=False)
                if extracted:
                    paper['trafilatura_content'] = extracted

            # Also parse HTML for specific fields if needed
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            # Get abstract from blockquote if not already present
            if not paper.get('abstract'):
                abstract_elem = soup.find('blockquote', class_='abstract')
                if abstract_elem:
                    abstract_text = abstract_elem.get_text(strip=True)
                    if abstract_text.startswith('Abstract:'):
                        abstract_text = abstract_text[9:].strip()
                    paper['abstract'] = abstract_text

            time.sleep(0.5)  # Be polite to the server

        except Exception as e:
            print(f"  Error scraping {abs_url}: {e}")

        return paper

    def screenshot_and_ocr(self, paper, use_ocr=False):
        """
        Take screenshot of abstract page and use Tesseract OCR to extract text
        This is a fallback method when other extraction fails
        """
        if not use_ocr:
            return paper

        abs_url = paper['url']
        print(f"  Using OCR for: {paper['title'][:60]}...")

        try:
            # Setup headless Chrome
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")

            driver = webdriver.Chrome(options=chrome_options)
            driver.get(abs_url)

            # Wait for page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "abstract"))
            )

            # Take screenshot
            screenshot = driver.get_screenshot_as_png()
            driver.quit()

            # Convert to PIL Image
            image = Image.open(io.BytesIO(screenshot))

            # Perform OCR
            ocr_text = pytesseract.image_to_string(image)
            paper['ocr_text'] = ocr_text

            # Try to extract abstract from OCR text
            if 'Abstract' in ocr_text and not paper.get('abstract'):
                lines = ocr_text.split('\n')
                abstract_started = False
                abstract_lines = []
                for line in lines:
                    if 'Abstract' in line:
                        abstract_started = True
                        continue
                    if abstract_started:
                        if line.strip() and not line.startswith('Submission'):
                            abstract_lines.append(line.strip())
                        elif len(abstract_lines) > 3:
                            break

                if abstract_lines:
                    paper['abstract_from_ocr'] = ' '.join(abstract_lines)

        except Exception as e:
            print(f"  Error with OCR for {abs_url}: {e}")

        return paper

    def scrape_category(self, category, max_results=200, use_ocr=False, use_trafilatura=True):
        """
        Complete scraping pipeline for a category
        Args:
            category: arXiv category (e.g., 'cs.CL')
            max_results: number of papers to fetch
            use_ocr: whether to use Tesseract OCR (slower, use as fallback)
            use_trafilatura: whether to use Trafilatura for content extraction
        """
        # Step 1: Fetch papers from API
        papers = self.fetch_papers(category, max_results)

        if not papers:
            print("No papers found!")
            return []

        # Step 2: Scrape individual pages if needed
        if use_trafilatura:
            for i, paper in enumerate(papers):
                print(f"Processing {i + 1}/{len(papers)}")
                papers[i] = self.scrape_abstract_page(paper)

                # Use OCR as fallback if abstract is still missing
                if use_ocr and not papers[i].get('abstract'):
                    papers[i] = self.screenshot_and_ocr(papers[i], use_ocr=True)

        # Step 3: Save results
        output_file = self.output_dir / f"{category.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.save_results(papers, output_file)

        return papers

    def save_results(self, papers, output_file):
        """Save papers to JSON file"""
        # Clean up papers to keep only required fields
        cleaned_papers = []
        for paper in papers:
            cleaned = {
                'url': paper.get('url', ''),
                'title': paper.get('title', ''),
                'abstract': paper.get('abstract', paper.get('abstract_from_ocr', '')),
                'authors': paper.get('authors', []),
                'date': paper.get('date', '')
            }
            cleaned_papers.append(cleaned)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_papers, f, indent=2, ensure_ascii=False)

        print(f"\nSaved {len(papers)} papers to {output_file}")
        print(f"Papers with abstracts: {sum(1 for p in cleaned_papers if p['abstract'])}")


def main():
    scraper = ArxivScraper()

    # Example: Scrape latest 200 papers from cs.CL (Computation and Language)
    category = "cs.CL"

    # Option 1: Fast scraping (API only, already includes abstracts)
    print("=== Fast Scraping (API only) ===")
    papers = scraper.scrape_category(category, max_results=200, use_ocr=False, use_trafilatura=False)

    # Option 2: Enhanced scraping with Trafilatura (slower but more complete)
    # print("=== Enhanced Scraping (with Trafilatura) ===")
    # papers = scraper.scrape_category(category, max_results=10, use_ocr=False, use_trafilatura=True)

    # Option 3: Full scraping with OCR fallback (slowest, most complete)
    # print("=== Full Scraping (with OCR) ===")
    # papers = scraper.scrape_category(category, max_results=5, use_ocr=True, use_trafilatura=True)

    # Display sample results
    if papers:
        print("\n=== Sample Paper ===")
        sample = papers[0]
        print(f"Title: {sample['title']}")
        print(f"Authors: {', '.join(sample['authors'][:3])}...")
        print(f"Date: {sample['date']}")
        print(f"URL: {sample['url']}")
        print(f"Abstract: {sample.get('abstract', 'N/A')[:200]}...")


if __name__ == "__main__":
    main()