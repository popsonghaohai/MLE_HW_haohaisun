# Installation Guide

This guide will walk you through setting up and running the Hybrid Search system.

### Prerequisites

*   **Python 3.8+**: Ensure you have a compatible version of Python installed.
*   **pip**: Python's package installer, usually comes with Python.
*   **Git** (Optional): For cloning the repository if it's hosted.

### Step-by-Step Installation

1.  **Clone the Repository (or Download Files)**
    If your project is in a Git repository, clone it:
    ```sh
    git clone <your-repository-url>
    cd HomeWork5
    ```
    If you've downloaded the files directly, navigate to the project's root directory:
    ```sh
    cd D:/AI_Project/PycharmProjects/PycharmProjects/HomeWork5
    ```

2.  **Create a Virtual Environment**
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```sh
    python -m venv .venv
    ```

3.  **Activate the Virtual Environment**
    *   **On Windows:**
        ```sh
        .venv\Scripts\activate
        ```
    *   **On macOS/Linux:**
        ```sh
        source .venv/bin/activate
        ```

4.  **Install Dependencies**
    Install all the required Python packages using `pip`:
    ```sh
    pip install fastapi uvicorn "sentence-transformers" "faiss-cpu" numpy
    ```
    *   **Note**: `faiss-cpu` is used for CPU-only FAISS. If you have a compatible GPU and want to use `faiss-gpu`, install `faiss-gpu` instead.

### Running the Application

1.  **Initial Data Loading and Evaluation**
    The first time you run the application, it needs to load data from `arxiv_data_rag/rag_data_2025Nov.json` into the SQLite database and build the FAISS index. This can take some time depending on the size of the JSON file and your system's performance.

    Run the `main.py` script directly:
    ```sh
    python main.py
    ```
    This script will:
    *   Create the `arxiv_data.db` SQLite database and its schema.
    *   Load data from `arxiv_data_rag/rag_data_2025Nov.json`.
    *   Generate embeddings and create `arxiv_faiss.index`.
    *   Run a basic evaluation of the search methods and print the results.
    *   Perform an example hybrid search.

    You should see output indicating the data loading progress and evaluation metrics.

2.  **Start the FastAPI Server**
    Once the data is loaded and indexed, you can start the FastAPI server:
    ```sh
    uvicorn main:app --reload
    ```
    The `--reload` flag enables auto-reloading of the server when code changes are detected (useful during development).

    You should see output similar to:
    ```
    INFO:     Will watch for changes in these directories: ['D:\\AI_Project\\PycharmProjects\\PycharmProjects\\HomeWork5']
    INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
    INFO:     Started reloader process [PID]
    INFO:     Started server process [PID]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    ```

### Using the API

The API will be available at `http://127.0.0.1:8000`.

**Hybrid Search Endpoint:**

*   **URL**: `http://127.0.0.1:8000/hybrid_search`
*   **Method**: `GET`
*   **Query Parameters**:
    *   `query` (string, required): The search query.
    *   `k` (integer, optional): The number of top results to return (default is 3).

**Example API Call (using `curl`):**

```sh
curl -X GET "http://localhost:8000/hybrid_search?query=transformer%20models%20for%20speech&k=2"
```

**Example API Call (using a web browser):**

Open your web browser and navigate to:
```
http://localhost:8000/hybrid_search?query=transformer%20models%20for%20speech&k=2
```

You can also access the interactive API documentation at `http://localhost:8000/docs` after starting the server.
