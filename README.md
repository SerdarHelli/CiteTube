# CiteTube - YouTube Transcript QA

CiteTube is a local application that allows you to ask questions about YouTube videos using their transcripts. The app fetches the transcript, chunks it, builds a vector index, and enables semantic search and question answering with proper citations.

## Features

- Fetch transcripts from YouTube videos
- Process and chunk transcripts for optimal retrieval
- Build FAISS vector index for semantic search
- Hybrid search combining vector similarity and BM25 keyword search
- Answer questions with proper timestamp citations
- User-friendly Gradio interface

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/CiteTube.git
cd CiteTube
```

2. Create a Python virtual environment:
```
python -m venv venv
```

3. Activate the virtual environment:
   - Windows:
   ```
   venv\Scripts\activate
   ```
   - macOS/Linux:
   ```
   source venv/bin/activate
   ```

4. Install the requirements:
```
pip install -r requirements.txt
```

5. Copy the example environment file and configure it:
```
copy .env.example .env
```

## Running the Application

### Step 1: Start the vLLM Server

You need to have a vLLM server running to use CiteTube. You can start one with:

```
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 --dtype float16
```

Alternatively, you can use other models like:
```
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --dtype float16
```

### Step 2: Run CiteTube

```
python app.py
```

This will start the Gradio interface. Open the provided URL in your browser (typically http://127.0.0.1:7860).

## Usage

1. **Ingest a YouTube Video**:
   - Paste a YouTube URL in the "Ingest Video" tab
   - Click "Ingest Video"
   - Wait for the ingestion process to complete

2. **Ask Questions**:
   - Switch to the "Ask Questions" tab
   - Type your question about the video content
   - Click "Ask"
   - View the answer with timestamp citations

## How It Works

1. **Ingestion Pipeline**:
   - Extract YouTube video ID from URL
   - Fetch transcript using youtube-transcript-api
   - Chunk transcript into segments
   - Embed segments using sentence-transformers (bge-m3)
   - Build FAISS index for vector search
   - Store metadata and segments in SQLite

2. **Retrieval Pipeline**:
   - Embed query using the same model
   - Perform vector search with FAISS
   - Perform keyword search with BM25
   - Combine results using Reciprocal Rank Fusion
   - Optionally rerank with a cross-encoder

3. **Answer Generation**:
   - Send query and retrieved segments to vLLM
   - Generate answer with proper citations
   - Return structured response with answer, bullets, and citations

## Project Structure

```
citetube/
├─ app.py                 # Gradio UI
├─ ingest.py              # fetch transcript, chunk, embed, build FAISS
├─ retrieve.py            # hybrid BM25 + FAISS + RRF + optional rerank
├─ llm.py                 # vLLM API client + prompt builder
├─ db.py                  # SQLite helpers
├─ models/                # embedding/reranker loaders
├─ data/
│  ├─ faiss/              # FAISS indexes
│  ├─ meta.db             # SQLite database
│  └─ logs/
├─ requirements.txt
├─ .env.example
└─ README.md
```

## Requirements

- Python 3.10+
- vLLM server (for LLM inference)
- Internet connection (for fetching YouTube transcripts)

## License

[MIT License](LICENSE)