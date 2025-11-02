import requests
import json
from pathlib import Path
import time
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import io
import concurrent.futures
from tqdm import tqdm
import re



class ArxivPDFOCR:
    def __init__(self, output_dir="arxiv_pdf_ocr"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.pdf_dir = self.output_dir / "pdfs"
        self.pdf_dir.mkdir(exist_ok=True)
        self.text_dir = self.output_dir / "extracted_text"
        self.text_dir.mkdir(exist_ok=True)
        self.images_dir = self.output_dir / "pdf_images"
        self.images_dir.mkdir(exist_ok=True)

    def load_papers_from_json(self, json_file):
        """Load papers from the JSON file created by Task 1"""
        with open(json_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        print(f"Loaded {len(papers)} papers from {json_file}")
        return papers

    def get_pdf_url(self, arxiv_url):
        """Convert arXiv abstract URL to PDF URL"""
        # Extract arXiv ID from URL (e.g., https://arxiv.org/abs/2301.12345)
        arxiv_id = arxiv_url.split('/abs/')[-1]
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        return pdf_url, arxiv_id

    def download_pdf(self, paper, force_redownload=False):
        """Download PDF for a paper"""
        pdf_url, arxiv_id = self.get_pdf_url(paper['url'])
        pdf_path = self.pdf_dir / f"{arxiv_id}.pdf"

        if pdf_path.exists() and not force_redownload:
            print(f"  PDF already exists: {pdf_path.name}")
            return pdf_path

        try:
            print(f"  Downloading: {pdf_url}")
            response = requests.get(pdf_url, timeout=30)

            if response.status_code == 200:
                with open(pdf_path, 'wb') as f:
                    f.write(response.content)
                print(f"  ✓ Downloaded: {pdf_path.name}")
                time.sleep(1)  # Be polite to arXiv servers
                return pdf_path
            else:
                print(f"  ✗ Error downloading {pdf_url}: {response.status_code}")
                return None

        except Exception as e:
            print(f"  ✗ Error downloading {pdf_url}: {e}")
            return None

    def pdf_to_images(self, pdf_path, dpi=300):
        """Convert PDF pages to images"""
        try:
            print(f"  Converting PDF to images: {pdf_path.name}")

            # Option 1: Specify poppler path for Windows
            poppler_path = r"C:\Program Files\poppler-25.07.0\Library\bin"  # Update this path

            # Check if poppler path exists
            import os
            if os.path.exists(poppler_path):
                images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
            else:
                # Try without specifying path (if it's in system PATH)
                images = convert_from_path(pdf_path, dpi=dpi)

            print(f"  ✓ Converted {len(images)} pages")
            return images
        except Exception as e:
            print(f"  ✗ Error converting PDF to images: {e}")
            print(f"  💡 Make sure Poppler is installed and in PATH")
            print(f"  💡 Windows: https://github.com/oschwartz10612/poppler-windows/releases/")
            return []

    def ocr_image(self, image, preserve_layout=True):
        """Perform OCR on a single image"""
        try:
            if preserve_layout:
                # Use PSM mode 6 for uniform block of text with layout preservation
                custom_config = r'--oem 3 --psm 6'
                text = pytesseract.image_to_string(image, config=custom_config)
            else:
                text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            print(f"  ✗ OCR Error: {e}")
            return ""

    def ocr_with_layout_detection(self, image):
        """
        Advanced OCR with layout detection
        Extracts text with position information to preserve structure
        """
        try:
            # Get detailed OCR data with bounding boxes
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

            # Group text by blocks and lines
            structured_text = []
            current_block = []
            prev_block_num = -1

            for i in range(len(ocr_data['text'])):
                if ocr_data['text'][i].strip():
                    block_num = ocr_data['block_num'][i]

                    # New block detected
                    if block_num != prev_block_num and current_block:
                        structured_text.append(' '.join(current_block))
                        structured_text.append('')  # Add blank line between blocks
                        current_block = []

                    current_block.append(ocr_data['text'][i])
                    prev_block_num = block_num

            # Add last block
            if current_block:
                structured_text.append(' '.join(current_block))

            return '\n'.join(structured_text)

        except Exception as e:
            print(f"  ✗ Layout detection error: {e}")
            return self.ocr_image(image, preserve_layout=True)

    def extract_sections(self, text):
        """
        Extract common paper sections (title, abstract, introduction, etc.)
        """
        sections = {
            'raw_text': text,
            'title': '',
            'abstract': '',
            'introduction': '',
            'conclusion': '',
            'references': ''
        }

        # Try to find title (usually first few lines, all caps or bold)
        lines = text.split('\n')
        title_lines = []
        for i, line in enumerate(lines[:10]):
            if line.strip() and len(line.strip()) > 10:
                title_lines.append(line.strip())
                if i > 0 and not lines[i + 1].strip():
                    break
        sections['title'] = ' '.join(title_lines)

        # Find abstract section
        abstract_match = re.search(r'Abstract\s*[:\-]?\s*(.+?)(?=\n\s*\n|\n[1-9]\.?\s+Introduction|$)',
                                   text, re.IGNORECASE | re.DOTALL)
        if abstract_match:
            sections['abstract'] = abstract_match.group(1).strip()

        # Find introduction
        intro_match = re.search(r'[1-9]\.?\s+Introduction\s*[:\-]?\s*(.+?)(?=\n\s*\n[1-9]\.?\s+\w+|$)',
                                text, re.IGNORECASE | re.DOTALL)
        if intro_match:
            sections['introduction'] = intro_match.group(1).strip()[:1000]  # First 1000 chars

        # Find conclusion
        conclusion_match = re.search(r'[1-9]\.?\s+Conclusion[s]?\s*[:\-]?\s*(.+?)(?=\n\s*\n[1-9]\.?\s+|References|$)',
                                     text, re.IGNORECASE | re.DOTALL)
        if conclusion_match:
            sections['conclusion'] = conclusion_match.group(1).strip()

        # Find references
        ref_match = re.search(r'References\s*[:\-]?\s*(.+?)$',
                              text, re.IGNORECASE | re.DOTALL)
        if ref_match:
            sections['references'] = ref_match.group(1).strip()[:2000]  # First 2000 chars

        return sections

    def ocr_pdf(self, pdf_path, paper_info, preserve_layout=True, save_images=False):
        """
        Complete OCR pipeline for a single PDF
        """
        arxiv_id = pdf_path.stem
        print(f"\n{'=' * 60}")
        print(f"Processing: {paper_info['title'][:50]}...")
        print(f"arXiv ID: {arxiv_id}")
        print(f"{'=' * 60}")

        # Convert PDF to images
        images = self.pdf_to_images(pdf_path)
        if not images:
            return None

        # Save images if requested
        if save_images:
            paper_image_dir = self.images_dir / arxiv_id
            paper_image_dir.mkdir(exist_ok=True)
            for i, img in enumerate(images):
                img.save(paper_image_dir / f"page_{i + 1:03d}.png")
            print(f"  Saved {len(images)} page images")

        # Perform OCR on each page
        full_text = []
        page_texts = []

        print(f"  Performing OCR on {len(images)} pages...")
        for i, image in enumerate(images, 1):
            print(f"    Page {i}/{len(images)}...", end='\r')

            if preserve_layout:
                page_text = self.ocr_with_layout_detection(image)
            else:
                page_text = self.ocr_image(image, preserve_layout=False)

            page_texts.append({
                'page_number': i,
                'text': page_text
            })
            full_text.append(f"\n{'=' * 50}\n PAGE {i} \n{'=' * 50}\n")
            full_text.append(page_text)

        print(f"    ✓ OCR completed for all pages{' ' * 20}")

        # Combine all text
        combined_text = '\n'.join(full_text)

        # Extract sections
        sections = self.extract_sections(combined_text)

        # Prepare result
        result = {
            'arxiv_id': arxiv_id,
            'url': paper_info['url'],
            'title': paper_info['title'],
            'authors': paper_info['authors'],
            'date': paper_info['date'],
            'ocr_metadata': {
                'num_pages': len(images),
                'preserve_layout': preserve_layout,
                'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'full_text': combined_text,
            'page_texts': page_texts,
            'extracted_sections': sections
        }

        # Save individual text file
        text_file = self.text_dir / f"{arxiv_id}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(f"Title: {paper_info['title']}\n")
            f.write(f"Authors: {', '.join(paper_info['authors'])}\n")
            f.write(f"Date: {paper_info['date']}\n")
            f.write(f"URL: {paper_info['url']}\n")
            f.write(f"\n{'=' * 80}\n\n")
            f.write(combined_text)

        print(f"  ✓ Saved text to: {text_file.name}")

        return result

    def process_single_paper(self, paper, download=True, preserve_layout=True, save_images=False):
        """Process a single paper: download PDF and perform OCR"""
        try:
            # Download PDF if needed
            if download:
                pdf_path = self.download_pdf(paper)
                if not pdf_path:
                    return None
            else:
                # Assume PDF already exists
                _, arxiv_id = self.get_pdf_url(paper['url'])
                pdf_path = self.pdf_dir / f"{arxiv_id}.pdf"
                if not pdf_path.exists():
                    print(f"  ✗ PDF not found: {pdf_path}")
                    return None

            # Perform OCR
            result = self.ocr_pdf(pdf_path, paper, preserve_layout, save_images)
            return result

        except Exception as e:
            print(f"  ✗ Error processing paper: {e}")
            return None

    def batch_process(self, papers, max_workers=3, download=True, preserve_layout=True,
                      save_images=False, max_papers=None):
        """
        Batch process multiple papers with parallel processing
        Args:
            papers: List of paper dictionaries
            max_workers: Number of parallel workers (be conservative to avoid overloading arXiv)
            download: Whether to download PDFs
            preserve_layout: Whether to preserve OCR layout
            save_images: Whether to save PDF page images
            max_papers: Limit number of papers to process (None for all)
        """
        if max_papers:
            papers = papers[:max_papers]

        results = []
        failed = []

        print(f"\n{'=' * 80}")
        print(f"Starting batch OCR processing for {len(papers)} papers")
        print(f"Workers: {max_workers}, Layout preservation: {preserve_layout}")
        print(f"{'=' * 80}\n")

        # Sequential processing (parallel can be added but be careful with arXiv rate limits)
        for i, paper in enumerate(papers, 1):
            print(f"\n[{i}/{len(papers)}] Processing: {paper['title'][:60]}...")

            result = self.process_single_paper(
                paper,
                download=download,
                preserve_layout=preserve_layout,
                save_images=save_images
            )

            if result:
                results.append(result)
            else:
                failed.append(paper)

            # Rate limiting for downloads
            if download and i < len(papers):
                time.sleep(3)  # 3 second delay between papers

        # Save combined results
        self.save_batch_results(results, failed)

        return results, failed

    def save_batch_results(self, results, failed):
        """Save batch processing results"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')

        # Save full results with all OCR data
        full_results_file = self.output_dir / f"ocr_results_full_{timestamp}.json"
        with open(full_results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Save summary results (without full text for easier viewing)
        summary_results = []
        for r in results:
            summary = {
                'arxiv_id': r['arxiv_id'],
                'url': r['url'],
                'title': r['title'],
                'authors': r['authors'],
                'date': r['date'],
                'num_pages': r['ocr_metadata']['num_pages'],
                'extracted_abstract': r['extracted_sections']['abstract'][:500] if r['extracted_sections'][
                    'abstract'] else '',
                'text_length': len(r['full_text'])
            }
            summary_results.append(summary)

        summary_file = self.output_dir / f"ocr_results_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False)

        # Save processing report
        report_file = self.output_dir / f"ocr_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"OCR Processing Report\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(f"Total papers processed: {len(results)}\n")
            f.write(f"Failed papers: {len(failed)}\n")
            f.write(f"Success rate: {len(results) / (len(results) + len(failed)) * 100:.1f}%\n\n")

            f.write(f"Total pages processed: {sum(r['ocr_metadata']['num_pages'] for r in results)}\n")
            f.write(f"Total text extracted: {sum(len(r['full_text']) for r in results):,} characters\n\n")

            if failed:
                f.write(f"\nFailed Papers:\n")
                f.write(f"{'-' * 80}\n")
                for paper in failed:
                    f.write(f"- {paper['title']}\n")
                    f.write(f"  URL: {paper['url']}\n\n")

        print(f"\n{'=' * 80}")
        print(f"✓ Batch processing completed!")
        print(f"  Full results: {full_results_file}")
        print(f"  Summary: {summary_file}")
        print(f"  Report: {report_file}")
        print(f"  Individual texts: {self.text_dir}/")
        print(f"\nSuccess: {len(results)}, Failed: {len(failed)}")
        print(f"{'=' * 80}\n")


def main():
    """Main execution function"""

    # Initialize OCR processor
    ocr_processor = ArxivPDFOCR(output_dir="arxiv_pdf_ocr")

    # Option 1: Load papers from Task 1 JSON file
    papers_json = "arxiv_data/cs_CL_20251028_100542.json"  # Update with your actual file
    # Uncomment to use:
    papers = ocr_processor.load_papers_from_json(papers_json)

    # Option 2: Create sample papers for testing
    #print("Creating sample papers for demonstration...")
   # papers = [
   #     {
   #         'url': 'https://arxiv.org/abs/2301.00001',
   #         'title': 'Sample Paper on Natural Language Processing',
  #          'authors': ['John Doe', 'Jane Smith'],
  #          'date': '2023-01-01',
  #          'abstract': 'This is a sample abstract.'
  #      },
  #      # Add more papers as needed
  #  ]

    # Process papers
    print("\n" + "=" * 80)
    print("ARXIV PDF BATCH OCR PROCESSOR")
    print("=" * 80)

    # Example 1: Process just a few papers for testing
   # print("\n>>> Processing sample papers...")
  #  results, failed = ocr_processor.batch_process(
  #      papers,
#   max_workers=2,
   #     download=True,
 #       preserve_layout=True,
#        save_images=True,  # Save page images for inspection
 #       max_papers=5  # Limit to 5 papers for testing
 #   )

    # Example 2: Process all papers (uncomment when ready)
    results, failed = ocr_processor.batch_process(
         papers,
         max_workers=2,
         download=True,
         preserve_layout=True,
         save_images=False,
         max_papers=None
     )

    # Display sample result
    if results:
        print("\n" + "=" * 80)
        print("SAMPLE EXTRACTED TEXT")
        print("=" * 80)
        sample = results[0]
        print(f"\nTitle: {sample['title']}")
        print(f"Pages: {sample['ocr_metadata']['num_pages']}")
        print(f"\nExtracted Abstract:")
        print(sample['extracted_sections']['abstract'][:500] if sample['extracted_sections']['abstract'] else "N/A")
        print(f"\nFirst 500 characters of full text:")
        print(sample['full_text'][:500])


if __name__ == "__main__":
    main()

