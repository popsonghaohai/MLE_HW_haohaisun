import requests
import json
import time
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
import fitz  # PyMuPDF
from transformers import AutoTokenizer
import logging
import pymupdf

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ArxivRAGPipeline:
    def __init__(self, output_dir="arxiv_data_rag"):
        self.base_url = "http://export.arxiv.org/api/query"
        self.output_dir = Path(output_dir)
        self.pdf_dir = self.output_dir / "pdfs"
        self.output_dir.mkdir(exist_ok=True)
        self.pdf_dir.mkdir(exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def fetch_papers(self, category, max_results=50):
        """Fetch papers from arXiv API for a given category."""
        params = {
            'search_query': f'cat:{category}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        logging.info(f"Fetching {max_results} papers from {category}...")
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching papers: {e}")
            return []

        return self._parse_arxiv_response(response.text)

    def _parse_arxiv_response(self, xml_content):
        """Parse arXiv API XML response."""
        root = ET.fromstring(xml_content)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        papers = []
        for entry in root.findall('atom:entry', namespace):
            paper = {}
            paper['url'] = entry.find('atom:id', namespace).text
            paper['arxiv_id'] = paper['url'].split('/abs/')[-1]
            paper['title'] = ' '.join(entry.find('atom:title', namespace).text.split())
            paper['abstract'] = ' '.join(entry.find('atom:summary', namespace).text.split())
            paper['authors'] = [author.find('atom:name', namespace).text for author in entry.findall('atom:author', namespace)]
            paper['date'] = entry.find('atom:published', namespace).text.split('T')[0]
            paper['pdf_url'] = f"https://arxiv.org/pdf/{paper['arxiv_id']}.pdf"
            papers.append(paper)
        logging.info(f"Parsed {len(papers)} papers from API.")
        return papers

    def download_pdf(self, paper):
        """Download PDF for a paper."""
        pdf_url = paper['pdf_url']
        pdf_filename = f"{paper['arxiv_id']}.pdf"
        pdf_path = self.pdf_dir / pdf_filename

        if pdf_path.exists():
            logging.info(f"PDF already exists: {pdf_path}")
            return str(pdf_path)

        logging.info(f"Downloading PDF: {pdf_url}")
        try:
            response = requests.get(pdf_url, timeout=20)
            response.raise_for_status()
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            time.sleep(1)  # Be polite
            return str(pdf_path)
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading {pdf_url}: {e}")
            return None

    def extract_text_from_pdf(self, pdf_path):
        """Extract full text from a PDF file."""
        if not pdf_path:
            return ""
        logging.info(f"Extracting text from: {pdf_path}")
        try:
            with fitz.open(pdf_path) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
            # Basic cleaning
            text = ' '.join(text.split())
            return text
        except Exception as e:
            logging.error(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def chunk_text(self, text, chunk_size=500, overlap=50):
        """Split text into overlapping chunks based on token count."""
        if not text:
            return []
        logging.info("Chunking text...")
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
        logging.info(f"Created {len(chunks)} chunks.")
        return chunks

    def save_processed_data(self, papers_data):
        """Save the final processed data to a JSON file."""
        output_file = self.output_dir / f"rag_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(papers_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved processed data for {len(papers_data)} papers to {output_file}")


def main():
    """Main function to run the RAG data preparation pipeline."""
    CATEGORY = "cs.CL"
    MAX_PAPERS = 50
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50

    pipeline = ArxivRAGPipeline()

    # 1. Data Collection: Fetch paper metadata
    papers = pipeline.fetch_papers(CATEGORY, max_results=MAX_PAPERS)

    processed_papers = []
    for i, paper in enumerate(papers):
        logging.info(f"Processing paper {i + 1}/{len(papers)}: {paper['title']}")

        # 2. Download PDF
        pdf_path = pipeline.download_pdf(paper)
        if not pdf_path:
            continue
        paper['pdf_path'] = pdf_path

        # 3. Text Extraction
        full_text = pipeline.extract_text_from_pdf(pdf_path)
        if not full_text:
            logging.warning(f"Could not extract text from {pdf_path}")
            paper['full_text'] = ""
            paper['text_chunks'] = []
        else:
            # 4. Text Chunking
            chunks = pipeline.chunk_text(full_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            paper['text_chunks'] = chunks
        
        processed_papers.append(paper)

    # 5. Save all processed data
    pipeline.save_processed_data(processed_papers)

    logging.info("RAG data preparation pipeline finished.")
    if processed_papers:
        logging.info(f"\n=== Sample of Processed Paper ===")
        sample = processed_papers[0]
        print(f"Title: {sample['title']}")
        print(f"URL: {sample['url']}")
        print(f"PDF Path: {sample['pdf_path']}")
        print(f"Number of chunks: {len(sample['text_chunks'])}")
        if sample['text_chunks']:
            print(f"First chunk sample: {sample['text_chunks'][0][:200]}...")


if __name__ == "__main__":
    main()
