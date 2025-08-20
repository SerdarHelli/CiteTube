# CiteTube

A local YouTube transcript QA application that uses Ollama for embeddings and language model inference. CiteTube allows you to ask questions about YouTube videos and get answers with precise timestamp citations.

## Features

- 🎥 **YouTube Transcript Ingestion**: Automatically fetch and process YouTube video transcripts
- 🔍 **Hybrid Search**: Combines pgvector similarity search with BM25 keyword search
- 🤖 **Local LLM**: Uses Ollama for both embeddings and language model inference
- 📝 **Precise Citations**: Every answer includes timestamp citations
- 🚀 **Fast Retrieval**: Efficient vector search with PostgreSQL + pgvector
- 🔄 **Reranking**: Optional cross-encoder reranking for improved results

## Architecture

CiteTube uses a modern RAG (Retrieval-Augmented Generation) architecture:

1. **Ingestion**: YouTube transcripts are fetched, chunked, and embedded using Ollama
2. **Storage**: Metadata and vectors stored in PostgreSQL with pgvector extension
3. **Retrieval**: Hybrid search combining semantic (pgvector) and keyword (BM25) search
4. **Generation**: Ollama LLM generates answers with timestamp citations

## Prerequisites

1. **Python 3.8+**
2. **Ollama**: Install from [https://ollama.ai/](https://ollama.ai/)
3. **PostgreSQL 12+** with **pgvector extension**

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd CiteTube
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .
   ```

4. **Install Ollama models**:
   ```bash
   python setup_ollama.py
   ```

   This will install:
   - `nomic-embed-text`: For text embeddings
   - `llama3.2`: For question answering

## Configuration

The application uses environment variables defined in `.env`:

```env
# LLM Configuration (Ollama)
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2

# Embedding Models (Ollama)
EMBEDDING_MODEL=nomic-embed-text

# Reranker Model (sentence-transformers)
RERANKER_MODEL=BAAI/bge-reranker-base

# Search Configuration
USE_RERANKER=true
TOP_K=8
TEMPERATURE=0.1
MAX_TOKENS=1024
```

### Alternative Models

You can use different Ollama models by updating the `.env` file:

**LLM Models**:
- `llama3.1`
- `mistral`
- `codellama`
- `phi3`

**Embedding Models**:
- `mxbai-embed-large`
- `all-minilm`

## Usage

### Command Line Interface

1. **Ingest a YouTube video**:
   ```python
   from citetube.ingestion.ingest import ingest_video
   
   video_id, metadata = ingest_video("https://www.youtube.com/watch?v=VIDEO_ID")
   print(f"Ingested video: {metadata['title']}")
   ```

2. **Ask questions**:
   ```python
   from citetube.retrieval.retrieve import hybrid_search
   from citetube.llm.llm import answer_question
   
   # Search for relevant segments
   segments = hybrid_search("What is machine learning?", video_id)
   
   # Get answer with citations
   response = answer_question("What is machine learning?", segments)
   print(response['answer'])
   ```

### Web Interface

Launch the Streamlit web interface:

```bash
python -m citetube.ui.app
```

## Testing

Test the Ollama integration:

```bash
python test_ollama.py
```

This will verify that both the embedding model and LLM are working correctly.

## Project Structure

```
CiteTube/
├── src/citetube/
│   ├── core/           # Core functionality
│   │   ├── config.py   # Configuration management
│   │   ├── db.py       # Database operations
│   │   └── models.py   # Ollama model wrappers
│   ├── ingestion/      # Video ingestion
│   │   └── ingest.py   # Transcript fetching and processing
│   ├── retrieval/      # Search and retrieval
│   │   └── retrieve.py # Hybrid search implementation
│   ├── llm/           # Language model interface
│   │   └── llm.py     # Ollama LLM client
│   └── ui/            # User interface
│       └── app.py     # Streamlit web app
├── data/              # Data storage
│   ├── faiss/         # FAISS indices
│   ├── logs/          # Application logs
│   └── meta.db        # SQLite database
├── test_ollama.py     # Integration tests
├── setup_ollama.py    # Model setup script
└── .env              # Environment configuration
```

## How It Works

### 1. Video Ingestion

```python
# Extract YouTube ID from URL
yt_id = extract_youtube_id(url)

# Fetch transcript using YouTube Transcript API
transcript_items, metadata = fetch_transcript(yt_id)

# Chunk transcript into overlapping segments
chunks = chunk_transcript(transcript_items)

# Generate embeddings using Ollama
embeddings = model.encode([chunk["text"] for chunk in chunks])

# Store in database and build FAISS index
store_segments(video_id, chunks)
build_faiss_index(video_id, segments)
```

### 2. Question Answering

```python
# Hybrid search: FAISS + BM25
faiss_results = search_faiss(query, video_id)
bm25_results = search_bm25(query, video_id)
combined_results = reciprocal_rank_fusion([faiss_results, bm25_results])

# Optional reranking
reranked_results = rerank_results(query, combined_results)

# Generate answer using Ollama
response = ollama_client.chat(
    model="llama3.2",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_prompt(question, segments)}
    ]
)
```

## Performance

- **Embedding Generation**: ~100-500 tokens/second (depends on model and hardware)
- **Search**: Sub-second retrieval for most queries
- **Answer Generation**: 10-50 tokens/second (depends on model and hardware)

## Troubleshooting

### Common Issues

1. **Ollama not found**:
   - Ensure Ollama is installed and in your PATH
   - Run `ollama --version` to verify installation

2. **Models not available**:
   - Run `python setup_ollama.py` to install required models
   - Check `ollama list` to see installed models

3. **Slow performance**:
   - Use smaller models (e.g., `llama3.2:1b` instead of `llama3.2`)
   - Reduce `MAX_TOKENS` in `.env`
   - Disable reranking by setting `USE_RERANKER=false`

4. **Memory issues**:
   - Use quantized models
   - Reduce batch sizes in embedding generation
   - Close other applications to free up RAM

### Logs

Check application logs in `data/logs/` for detailed error information:
- `ingest.log`: Video ingestion logs
- `retrieve.log`: Search and retrieval logs
- `llm.log`: Language model logs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM inference
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api) for transcript fetching
- [Streamlit](https://streamlit.io/) for the web interface