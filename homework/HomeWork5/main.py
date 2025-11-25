import sqlite3
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
import logging
import fastapi
import uvicorn
import sqlite3

import numpy
# --- Configuration ---
DB_PATH = "arxiv_data.db"
JSON_PATH = "./arxiv_data_rag/rag_data_2025Nov.json"
FAISS_INDEX_PATH = "arxiv_faiss.index"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
DIMENSION = 384

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- FastAPI App ---
app = FastAPI()

# --- Database and Search Index Class ---
class ArxivSearch:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None

    def create_schema(self):
        """Creates the SQLite database schema."""
        c = self.conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                arxiv_id TEXT UNIQUE,
                title TEXT,
                authors TEXT,
                year INTEGER
            )
        ''')
        c.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks USING fts5(
                content,
                doc_id,
                chunk_id
            )
        ''')
        self.conn.commit()

    def load_data(self):
        """Loads data from the JSON file into the database and creates the FAISS index."""
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        c = self.conn.cursor()
        embeddings = []
        chunk_texts = []

        for paper in data:
            try:
                c.execute("INSERT INTO documents (arxiv_id, title, authors, year) VALUES (?, ?, ?, ?)",
                          (paper['arxiv_id'], paper['title'], json.dumps(paper['authors']), int(paper['date'][:4])))
                doc_id = c.lastrowid
                for i, chunk in enumerate(paper['text_chunks']):
                    c.execute("INSERT INTO doc_chunks (doc_id, chunk_id, content) VALUES (?, ?, ?)",
                              (doc_id, i, chunk))
                    chunk_texts.append(chunk)
            except sqlite3.IntegrityError:
                logging.warning(f"Paper with arxiv_id {paper['arxiv_id']} already exists. Skipping.")

        self.conn.commit()

        if chunk_texts:
            logging.info("Generating embeddings for FAISS index...")
            embeddings = self.model.encode(chunk_texts, show_progress_bar=True)
            self.index = faiss.IndexFlatL2(DIMENSION)
            self.index.add(np.array(embeddings))
            faiss.write_index(self.index, FAISS_INDEX_PATH)
            logging.info(f"FAISS index created and saved to {FAISS_INDEX_PATH}")

    def keyword_search(self, query, k=5):
        """Performs a keyword search using FTS5."""
        c = self.conn.cursor()
        c.execute("""
            SELECT documents.doc_id, documents.title, doc_chunks.content, rank
            FROM doc_chunks
            JOIN documents ON doc_chunks.doc_id = documents.doc_id
            WHERE doc_chunks MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, k))
        return c.fetchall()

    def vector_search(self, query, k=5):
        """Performs a vector search using FAISS."""
        if self.index is None:
            try:
                self.index = faiss.read_index(FAISS_INDEX_PATH)
            except RuntimeError:
                logging.error("FAISS index not found. Please run the data loading process first.")
                return []
        
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        c = self.conn.cursor()
        for i in range(k):
            # The index in FAISS corresponds to the order of chunks inserted.
            # We need to retrieve the doc_id and content based on this index.
            # This is a simplification. A more robust solution would store a mapping.
            # For this example, we'll retrieve by rowid, assuming it matches.
            c.execute("""
                SELECT documents.doc_id, documents.title, doc_chunks.content
                FROM doc_chunks
                JOIN documents ON doc_chunks.doc_id = documents.doc_id
                WHERE doc_chunks.rowid = ?
            """, (int(indices[0][i]) + 1,)) # rowid is 1-based
            result = c.fetchone()
            if result:
                results.append(result + (distances[0][i],))
        return results

    def hybrid_search(self, query, k=5, alpha=0.5):
        """Performs a hybrid search combining keyword and vector search."""
        keyword_results = self.keyword_search(query, k)
        vector_results = self.vector_search(query, k)

        # Simple RRF-like scoring
        ranked_list = {}
        for i, (doc_id, _, content, _) in enumerate(keyword_results):
            if (doc_id, content) not in ranked_list:
                ranked_list[(doc_id, content)] = 0
            ranked_list[(doc_id, content)] += 1 / (i + 1)

        for i, (doc_id, _, content, _) in enumerate(vector_results):
            if (doc_id, content) not in ranked_list:
                ranked_list[(doc_id, content)] = 0
            ranked_list[(doc_id, content)] += alpha * (1 / (i + 1))

        sorted_results = sorted(ranked_list.items(), key=lambda item: item[1], reverse=True)
        
        final_results = []
        for (doc_id, content), score in sorted_results[:k]:
            c = self.conn.cursor()
            c.execute("SELECT title FROM documents WHERE doc_id = ?", (doc_id,))
            title = c.fetchone()[0]
            final_results.append({"doc_id": doc_id, "title": title, "content": content, "score": score})
            
        return final_results

# --- FastAPI Endpoints ---
search_system = ArxivSearch()

@app.on_event("startup")
def startup_event():
    search_system.create_schema()
    # Check if data is already loaded
    c = search_system.conn.cursor()
    c.execute("SELECT COUNT(*) FROM documents")
    if c.fetchone()[0] == 0:
        logging.info("Database is empty. Loading data...")
        search_system.load_data()
    else:
        logging.info("Database already contains data. Skipping data loading.")
        # Load FAISS index if it exists
        try:
            search_system.index = faiss.read_index(FAISS_INDEX_PATH)
            logging.info("FAISS index loaded.")
        except RuntimeError:
            logging.warning("FAISS index not found. Vector search will not be available until data is re-loaded.")


@app.get("/hybrid_search")
async def hybrid_search_endpoint(query: str, k: int = 3):
    """Performs hybrid search and returns the top-k results."""
    results = search_system.hybrid_search(query, k)
    return {"results": results}

# --- Evaluation ---
def evaluate_search():
    """Evaluates the performance of different search methods."""
    test_queries = [
        {"query": "transformer models for speech", "relevant_doc_ids": [3]},
        {"query": "large language model security", "relevant_doc_ids": [4]},
        {"query": "reinforcement learning for robotics", "relevant_doc_ids": [10]},
        {"query": "multi-agent systems", "relevant_doc_ids": [4, 11]},
        {"query": "sentiment analysis in social media", "relevant_doc_ids": [12]},
        {"query": "knowledge distillation in NLP", "relevant_doc_ids": [13]},
        {"query": "vision-language models", "relevant_doc_ids": [13, 14]},
        {"query": "causal inference in machine learning", "relevant_doc_ids": [10]},
        {"query": "graph neural networks for heterogeneous graphs", "relevant_doc_ids": [15]},
        {"query": "conformal prediction for uncertainty", "relevant_doc_ids": [16]}
    ]
    
    k = 3
    metrics = {"keyword": {"recall@3": 0}, "vector": {"recall@3": 0}, "hybrid": {"recall@3": 0}}

    for item in test_queries:
        query = item["query"]
        relevant_ids = set(item["relevant_doc_ids"])

        # Keyword search
        keyword_results = search_system.keyword_search(query, k)
        retrieved_ids = {res[0] for res in keyword_results}
        if relevant_ids.intersection(retrieved_ids):
            metrics["keyword"]["recall@3"] += 1

        # Vector search
        vector_results = search_system.vector_search(query, k)
        retrieved_ids = {res[0] for res in vector_results}
        if relevant_ids.intersection(retrieved_ids):
            metrics["vector"]["recall@3"] += 1

        # Hybrid search
        hybrid_results = search_system.hybrid_search(query, k)
        retrieved_ids = {res["doc_id"] for res in hybrid_results}
        if relevant_ids.intersection(retrieved_ids):
            metrics["hybrid"]["recall@3"] += 1

    num_queries = len(test_queries)
    for method in metrics:
        metrics[method]["recall@3"] /= num_queries

    logging.info(f"Evaluation Metrics: {metrics}")
    return metrics

if __name__ == "__main__":
    # This part is for command-line execution, e.g., for initial data loading and evaluation.
    # To run the FastAPI server, use: uvicorn main:app --reload
    
    # Create schema and load data if db is empty
    search_system.create_schema()
    c = search_system.conn.cursor()
    c.execute("SELECT COUNT(*) FROM documents")
    if c.fetchone()[0] == 0:
        logging.info("Database is empty. Loading data...")
        search_system.load_data()
    else:
        logging.info("Database already contains data.")

    # Run evaluation
    evaluate_search()

    # Example search
    print("\n--- Example Hybrid Search ---")
    example_query = "What are the risks of LLM-based multi-agent systems?"
    results = search_system.hybrid_search(example_query, k=3)
    for res in results:
        print(f"Title: {res['title']}\nContent: {res['content'][:200]}...\nScore: {res['score']}\n")

