# Installation Guide

This document provides instructions on how to set up the environment and install the necessary dependencies for the arXiv RAG pipeline project.

## 1. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies and avoid conflicts with other Python projects.

### Using `venv`

Open your terminal or command prompt and run the following commands:

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# On Windows
venv\\Scripts\\activate

# On macOS and Linux
source venv/bin/activate
```

## 2. Install Dependencies

With the virtual environment activated, install the required packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Important Note on `PyMuPDF`

The script uses the `fitz` module to extract text from PDF files. This module is part of the `PyMuPDF` library. The `requirements.txt` file correctly lists `PyMuPDF` as a dependency, so `pip` will handle the installation automatically.

If you encounter an error like `module 'fitz' has no attribute 'open'`, it likely means that the `PyMuPDF` library was not installed correctly. Ensure you have installed the packages from the `requirements.txt` file as shown above.

## 3. Running the Script

Once the dependencies are installed, you can run the main script to start the data preparation pipeline:

```bash
python main.py
```

This will:
1.  Fetch paper metadata from arXiv.
2.  Download the PDFs into the `arxiv_data_rag/pdfs` directory.
3.  Extract text from the PDFs.
4.  Chunk the text into smaller segments.
5.  Save the processed data into a JSON file in the `arxiv_data_rag` directory.
