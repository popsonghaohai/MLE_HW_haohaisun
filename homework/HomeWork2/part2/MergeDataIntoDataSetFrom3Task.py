"""
End-to-End Dataset Cleaner
Merges outputs from:
- Scrape_arXiv_200papers.py (JSON with paper metadata)
- Scrape_arXiv_PDF.py (JSON with extracted PDF text)
- WhisterYoutubeTransBot.py (JSONL with video transcripts)

Cleaning Pipeline:
1. Language detection (langdetect)
2. Strip HTML noise
3. MinHash deduplication (similarity ≥ 0.7)
4. Remove PII (emails, credit cards, phone numbers)
5. Remove repetitive n-grams
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
import html

# Required packages
from langdetect import detect, LangDetectException  # pip install langdetect
from datasketch import MinHash, MinHashLSH  # pip install datasketch


class DatasetCleaner:
    def __init__(self, output_dir="cleaned_dataset"):
        """
        Initialize Dataset Cleaner

        Args:
            output_dir: Output directory for cleaned dataset
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # PII patterns
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
        self.credit_card_pattern = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')
        self.ssn_pattern = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')

        # HTML patterns
        self.html_tag_pattern = re.compile(r'<[^>]+>')
        self.html_entity_pattern = re.compile(r'&[a-zA-Z]+;|&#\d+;')

        # Stats
        self.stats = {
            'total_documents': 0,
            'arxiv_papers': 0,
            'arxiv_pdfs': 0,
            'youtube_transcripts': 0,
            'non_english_filtered': 0,
            'duplicates_removed': 0,
            'pii_removed': 0,
            'repetitive_removed': 0,
            'final_documents': 0
        }

    def load_arxiv_papers(self, json_path: Path) -> List[Dict]:
        """
        Load arXiv paper metadata from Scrape_arXiv_200papers.py output

        Expected format: List of dicts with 'title', 'abstract', 'authors', etc.
        """
        print(f"\nLoading arXiv papers: {json_path}")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            documents = []
            if isinstance(data, list):
                for item in data:
                    doc = {
                        'id': item.get('id', item.get('arxiv_id', f"arxiv_{len(documents)}")),
                        'source': 'arxiv_metadata',
                        'title': item.get('title', ''),
                        'text': f"{item.get('title', '')} {item.get('abstract', '')}",
                        'metadata': {
                            'authors': item.get('authors', []),
                            'published': item.get('published', ''),
                            'url': item.get('url', ''),
                            'categories': item.get('categories', [])
                        }
                    }
                    documents.append(doc)

            print(f"  ✓ Loaded {len(documents)} arXiv papers")
            self.stats['arxiv_papers'] = len(documents)
            return documents

        except Exception as e:
            print(f"  ✗ Error loading arXiv papers: {e}")
            return []

    def load_arxiv_pdfs(self, json_path: Path) -> List[Dict]:
        """
        Load arXiv PDF text from Scrape_arXiv_PDF.py output

        Expected format: Dict or list with extracted PDF text
        """
        print(f"\nLoading arXiv PDFs: {json_path}")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            documents = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get('papers', [data])
            else:
                items = []

            for item in items:
                doc = {
                    'id': item.get('id', item.get('arxiv_id', f"pdf_{len(documents)}")),
                    'source': 'arxiv_pdf',
                    'title': item.get('title', ''),
                    'text': item.get('text', item.get('content', '')),
                    'metadata': {
                        'url': item.get('url', ''),
                        'pages': item.get('pages', 0)
                    }
                }
                documents.append(doc)

            print(f"  ✓ Loaded {len(documents)} arXiv PDFs")
            self.stats['arxiv_pdfs'] = len(documents)
            return documents

        except Exception as e:
            print(f"  ✗ Error loading arXiv PDFs: {e}")
            return []

    def load_youtube_transcripts(self, jsonl_path: Path) -> List[Dict]:
        """
        Load YouTube transcripts from WhisterYoutubeTransBot.py output

        Expected format: JSONL with entries per line
        """
        print(f"\nLoading YouTube transcripts: {jsonl_path}")

        try:
            # Group entries by video
            video_texts = defaultdict(list)

            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    video_id = jsonl_path.stem  # Use filename as video ID
                    video_texts[video_id].append(entry.get('text', ''))

            # Create documents
            documents = []
            for video_id, texts in video_texts.items():
                doc = {
                    'id': video_id,
                    'source': 'youtube_transcript',
                    'title': f"Video: {video_id}",
                    'text': ' '.join(texts),
                    'metadata': {
                        'segments': len(texts)
                    }
                }
                documents.append(doc)

            print(f"  ✓ Loaded {len(documents)} YouTube transcripts")
            self.stats['youtube_transcripts'] = len(documents)
            return documents

        except Exception as e:
            print(f"  ✗ Error loading YouTube transcripts: {e}")
            return []

    def load_all_sources(self, arxiv_papers_dir: str, arxiv_pdfs_dir: str,
                        youtube_dir: str) -> List[Dict]:
        """
        Load documents from all sources

        Args:
            arxiv_papers_dir: Directory with arXiv paper JSONs
            arxiv_pdfs_dir: Directory with arXiv PDF JSONs
            youtube_dir: Directory with YouTube JSONL files
        """
        print("\n" + "=" * 80)
        print("LOADING DATA FROM ALL SOURCES")
        print("=" * 80)

        documents = []

        # Load arXiv papers
        arxiv_papers_path = Path(arxiv_papers_dir)
        if arxiv_papers_path.exists():
            for json_file in arxiv_papers_path.glob("*.json"):
                documents.extend(self.load_arxiv_papers(json_file))

        # Load arXiv PDFs
        arxiv_pdfs_path = Path(arxiv_pdfs_dir)
        if arxiv_pdfs_path.exists():
            for json_file in arxiv_pdfs_path.glob("*.json"):
                documents.extend(self.load_arxiv_pdfs(json_file))

        # Load YouTube transcripts
        youtube_path = Path(youtube_dir)
        if youtube_path.exists():
            for jsonl_file in youtube_path.glob("*.jsonl"):
                documents.extend(self.load_youtube_transcripts(jsonl_file))

        self.stats['total_documents'] = len(documents)
        print(f"\n✓ Total documents loaded: {len(documents)}")
        print(f"  - arXiv papers: {self.stats['arxiv_papers']}")
        print(f"  - arXiv PDFs: {self.stats['arxiv_pdfs']}")
        print(f"  - YouTube transcripts: {self.stats['youtube_transcripts']}")

        return documents

    def detect_language(self, text: str) -> str:
        """
        Detect language of text using langdetect

        Returns:
            Language code (e.g., 'en', 'es', 'fr') or 'unknown'
        """
        try:
            if not text or len(text.strip()) < 10:
                return 'unknown'
            return detect(text)
        except LangDetectException:
            return 'unknown'

    def filter_by_language(self, documents: List[Dict], target_lang='en') -> List[Dict]:
        """
        Filter documents by language

        Args:
            documents: List of documents
            target_lang: Target language code (default: 'en')
        """
        print(f"\n{'=' * 80}")
        print(f"FILTERING BY LANGUAGE ({target_lang})")
        print(f"{'=' * 80}")

        filtered = []
        for doc in documents:
            lang = self.detect_language(doc['text'])
            doc['metadata']['language'] = lang

            if lang == target_lang:
                filtered.append(doc)
            else:
                self.stats['non_english_filtered'] += 1

        print(f"  ✓ Kept {len(filtered)} documents")
        print(f"  ✗ Filtered {self.stats['non_english_filtered']} non-{target_lang} documents")

        return filtered

    def strip_html(self, text: str) -> str:
        """
        Remove HTML tags and entities from text
        """
        # Decode HTML entities
        text = html.unescape(text)

        # Remove HTML tags
        text = self.html_tag_pattern.sub('', text)

        # Remove remaining HTML entities
        text = self.html_entity_pattern.sub('', text)

        return text

    def clean_html_noise(self, documents: List[Dict]) -> List[Dict]:
        """
        Strip HTML noise from all documents
        """
        print(f"\n{'=' * 80}")
        print("CLEANING HTML NOISE")
        print(f"{'=' * 80}")

        for doc in documents:
            doc['text'] = self.strip_html(doc['text'])
            doc['title'] = self.strip_html(doc['title'])

        print(f"  ✓ Cleaned HTML from {len(documents)} documents")

        return documents

    def create_minhash(self, text: str, num_perm=128) -> MinHash:
        """
        Create MinHash signature for text

        Args:
            text: Input text
            num_perm: Number of permutations (higher = more accurate)
        """
        minhash = MinHash(num_perm=num_perm)

        # Tokenize into shingles (3-grams of words)
        words = text.lower().split()
        for i in range(len(words) - 2):
            shingle = ' '.join(words[i:i+3])
            minhash.update(shingle.encode('utf-8'))

        return minhash

    def deduplicate_with_minhash(self, documents: List[Dict],
                                 threshold=0.7, num_perm=128) -> List[Dict]:
        """
        Remove duplicate documents using MinHash LSH

        Args:
            documents: List of documents
            threshold: Jaccard similarity threshold (0.7 = 70% similar)
            num_perm: Number of permutations for MinHash
        """
        print(f"\n{'=' * 80}")
        print(f"DEDUPLICATING WITH MINHASH (threshold={threshold})")
        print(f"{'=' * 80}")

        # Create LSH index
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

        # Track unique documents
        unique_docs = []
        seen_ids = set()

        for i, doc in enumerate(documents):
            doc_id = f"{doc['source']}_{doc['id']}"

            # Create MinHash signature
            minhash = self.create_minhash(doc['text'], num_perm)

            # Query LSH for similar documents
            similar = lsh.query(minhash)

            if not similar:
                # No duplicates found - add to unique set
                lsh.insert(doc_id, minhash)
                unique_docs.append(doc)
                seen_ids.add(doc_id)
            else:
                # Duplicate found
                self.stats['duplicates_removed'] += 1

        print(f"  ✓ Kept {len(unique_docs)} unique documents")
        print(f"  ✗ Removed {self.stats['duplicates_removed']} duplicates")

        return unique_docs

    def remove_pii(self, text: str) -> Tuple[str, int]:
        """
        Remove PII from text

        Returns:
            (cleaned_text, num_removals)
        """
        removals = 0

        # Remove emails
        matches = len(self.email_pattern.findall(text))
        text = self.email_pattern.sub('[EMAIL]', text)
        removals += matches

        # Remove phone numbers
        matches = len(self.phone_pattern.findall(text))
        text = self.phone_pattern.sub('[PHONE]', text)
        removals += matches

        # Remove credit card numbers
        matches = len(self.credit_card_pattern.findall(text))
        text = self.credit_card_pattern.sub('[CREDIT_CARD]', text)
        removals += matches

        # Remove SSN
        matches = len(self.ssn_pattern.findall(text))
        text = self.ssn_pattern.sub('[SSN]', text)
        removals += matches

        return text, removals

    def clean_pii(self, documents: List[Dict]) -> List[Dict]:
        """
        Remove PII from all documents
        """
        print(f"\n{'=' * 80}")
        print("REMOVING PII")
        print(f"{'=' * 80}")

        total_removals = 0

        for doc in documents:
            cleaned_text, removals = self.remove_pii(doc['text'])
            doc['text'] = cleaned_text
            total_removals += removals

            if removals > 0:
                self.stats['pii_removed'] += 1

        print(f"  ✓ Removed PII from {self.stats['pii_removed']} documents")
        print(f"  ✓ Total PII instances: {total_removals}")

        return documents

    def find_repetitive_ngrams(self, text: str, n=3, threshold=3) -> Set[str]:
        """
        Find n-grams that repeat more than threshold times

        Args:
            text: Input text
            n: N-gram size
            threshold: Minimum repetitions to be considered repetitive
        """
        words = text.lower().split()
        ngrams = []

        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)

        # Count n-grams
        ngram_counts = Counter(ngrams)

        # Find repetitive ones
        repetitive = {ngram for ngram, count in ngram_counts.items()
                     if count >= threshold}

        return repetitive

    def remove_repetitive_ngrams(self, documents: List[Dict],
                                n=5, threshold=3) -> List[Dict]:
        """
        Remove documents with excessive repetitive n-grams

        Args:
            documents: List of documents
            n: N-gram size
            threshold: Minimum repetitions to flag as repetitive
        """
        print(f"\n{'=' * 80}")
        print(f"REMOVING REPETITIVE N-GRAMS (n={n}, threshold={threshold})")
        print(f"{'=' * 80}")

        filtered = []

        for doc in documents:
            repetitive = self.find_repetitive_ngrams(doc['text'], n, threshold)

            # Calculate repetition ratio
            words = doc['text'].split()
            if len(words) < 10:
                continue

            repetition_ratio = len(repetitive) / max(len(words) - n + 1, 1)

            # Keep if not too repetitive
            if repetition_ratio < 0.3:  # Less than 30% repetitive
                filtered.append(doc)
            else:
                self.stats['repetitive_removed'] += 1

        print(f"  ✓ Kept {len(filtered)} documents")
        print(f"  ✗ Removed {self.stats['repetitive_removed']} repetitive documents")

        return filtered

    def save_cleaned_dataset(self, documents: List[Dict], filename="cleaned_dataset.json"):
        """
        Save cleaned dataset to JSON file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Saved cleaned dataset: {output_path}")
        print(f"  Total documents: {len(documents)}")

        return output_path

    def save_stats(self, filename="cleaning_stats.json"):
        """
        Save cleaning statistics
        """
        stats_path = self.output_dir / filename

        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)

        print(f"✓ Saved statistics: {stats_path}")

        return stats_path

    def clean_pipeline(self, arxiv_papers_dir: str, arxiv_pdfs_dir: str,
                      youtube_dir: str, target_lang='en',
                      dedup_threshold=0.7) -> List[Dict]:
        """
        Complete end-to-end cleaning pipeline

        Args:
            arxiv_papers_dir: Directory with arXiv paper JSONs
            arxiv_pdfs_dir: Directory with arXiv PDF JSONs
            youtube_dir: Directory with YouTube JSONL files
            target_lang: Target language for filtering
            dedup_threshold: MinHash similarity threshold
        """
        print("\n" + "=" * 80)
        print("END-TO-END DATASET CLEANING PIPELINE")
        print("=" * 80)

        # Step 1: Load all sources
        documents = self.load_all_sources(arxiv_papers_dir, arxiv_pdfs_dir, youtube_dir)

        if not documents:
            print("\n✗ No documents loaded. Exiting.")
            return []

        # Step 2: Filter by language
        documents = self.filter_by_language(documents, target_lang)

        # Step 3: Strip HTML noise
        documents = self.clean_html_noise(documents)

        # Step 4: MinHash deduplication
        documents = self.deduplicate_with_minhash(documents, threshold=dedup_threshold)

        # Step 5: Remove PII
        documents = self.clean_pii(documents)

        # Step 6: Remove repetitive n-grams
        documents = self.remove_repetitive_ngrams(documents)

        self.stats['final_documents'] = len(documents)

        # Save results
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)

        self.save_cleaned_dataset(documents)
        self.save_stats()

        # Print summary
        self.print_summary()

        return documents

    def print_summary(self):
        """
        Print cleaning pipeline summary
        """
        print("\n" + "=" * 80)
        print("CLEANING PIPELINE SUMMARY")
        print("=" * 80)

        print(f"\nInput:")
        print(f"  Total documents: {self.stats['total_documents']}")
        print(f"  - arXiv papers: {self.stats['arxiv_papers']}")
        print(f"  - arXiv PDFs: {self.stats['arxiv_pdfs']}")
        print(f"  - YouTube transcripts: {self.stats['youtube_transcripts']}")

        print(f"\nFiltering:")
        print(f"  Non-English filtered: {self.stats['non_english_filtered']}")
        print(f"  Duplicates removed: {self.stats['duplicates_removed']}")
        print(f"  PII cleaned: {self.stats['pii_removed']}")
        print(f"  Repetitive removed: {self.stats['repetitive_removed']}")

        print(f"\nOutput:")
        print(f"  Final documents: {self.stats['final_documents']}")

        retention_rate = (self.stats['final_documents'] /
                         max(self.stats['total_documents'], 1)) * 100
        print(f"  Retention rate: {retention_rate:.1f}%")

        print("=" * 80 + "\n")


def main():
    """Example usage"""

    # Initialize cleaner
    cleaner = DatasetCleaner(output_dir="cleaned_dataset")

    # Run cleaning pipeline
    cleaned_documents = cleaner.clean_pipeline(
        arxiv_papers_dir="arxiv_data",
        arxiv_pdfs_dir="arxiv_pdf_ocr",
        youtube_dir="nlp_conference_transcripts",
        target_lang='en',
        dedup_threshold=0.7
    )

    print(f"\n✓ Cleaning complete!")
    print(f"✓ Cleaned {len(cleaned_documents)} documents")
    print(f"✓ Output: {cleaner.output_dir}")


if __name__ == "__main__":
    main()