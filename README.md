# CiteTube

A modern local YouTube transcript QA application that uses vLLM for language model inference and sentence-transformers for embeddings. CiteTube allows you to ask questions about YouTube videos and get answers with precise timestamp citations.

## Features

- 🎥 **YouTube Transcript Ingestion**: Automatically fetch and process YouTube video transcripts
- 🔍 **Hybrid Search**: Combines pgvector similarity search with BM25 keyword search
- 🤖 **Local LLM**: Uses vLLM for high-performance local language model inference
- 📝 **Precise Citations**: Every answer includes timestamp citations
- 🚀 **Fast Retrieval**: Efficient vector search with PostgreSQL + pgvector
- 🔄 **Reranking**: Optional cross-encoder reranking for improved results
- 🐳 **Docker Support**: Easy deployment with Docker and Docker Compose
- 🖥️ **Modern CLI**: Rich command-line interface with typer and rich
- 🛠️ **Developer Tools**: Integrated linting, formatting, and testing

## Architecture

CiteTube uses a modern RAG (Retrieval-Augmented Generation) architecture:

1. **Ingestion**: YouTube transcripts are fetched, chunked, and embedded using sentence-transformers
2. **Storage**: Metadata and vectors stored in PostgreSQL with pgvector extension
3. **Retrieval**: Hybrid search combining semantic (pgvector) and keyword (BM25) search
4. **Generation**: vLLM serves local LLM to generate answers with timestamp citations

## Quick Start

### Option 1: Smart Mode (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd CiteTube

# Copy and configure environment
cp .env.example .env
# Edit .env with your preferred settings

# Install dependencies
pip install -r requirements.txt

# Initialize database
python main.py init

# Smart start - handles everything automatically!
python main.py run
```

The smart `run` command automatically:
- ✅ Checks and creates .env file if needed
- ✅ Ensures directories exist
- ✅ Tests database connection
- ✅ Starts vLLM if not running
- ✅ Launches the web application

### Option 2: Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database
python main.py init

# Smart start (recommended)
python main.py run

# Or start all services explicitly
python main.py start
```

### Option 3: Docker (Production)

```bash
# Start with Docker Compose
docker-compose -f docker/docker-compose.yml up -d

# Check status
docker-compose -f docker/docker-compose.yml ps
```

## Installation

### Prerequisites

- **Python 3.8+**
- **CUDA-compatible GPU** (recommended for optimal performance)
- **8GB+ RAM** (16GB+ recommended for larger models)
- **PostgreSQL 12+** with **pgvector extension**

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y postgresql postgresql-contrib postgresql-server-dev-14 \
                    build-essential git curl python3-pip python3-venv
```

**PostgreSQL Setup:**
```bash
sudo systemctl start postgresql
sudo systemctl enable postgresql
sudo -u postgres psql -c "CREATE DATABASE citetube;"
sudo -u postgres psql -c "ALTER USER postgres PASSWORD '12345';"

# Install pgvector extension
cd /tmp
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
sudo -u postgres psql -d citetube -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Python Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install vLLM with CUDA support
pip install -U vllm --torch-backend=cu128  # For CUDA 12.8+
# pip install -U vllm --torch-backend=cu121  # For CUDA 12.1+
# pip install -U vllm --torch-backend=cu118  # For CUDA 11.8
```

## Configuration

The application uses environment variables defined in `.env`:

```env
# vLLM Configuration
VLLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct  # Small model for testing
VLLM_HOST=localhost
VLLM_PORT=8000
VLLM_MAX_MODEL_LEN=8192
VLLM_GPU_MEMORY_UTILIZATION=0.85

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=citetube
DB_USER=postgres
DB_PASSWORD=12345

# Embedding Models
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=BAAI/bge-reranker-base

# Search Configuration
USE_RERANKER=true
TOP_K=8
TEMPERATURE=0.1
MAX_TOKENS=1024
```

### Recommended Models

**For Testing/Development:**
- `Qwen/Qwen2.5-0.5B-Instruct` (fast, low memory)
- `Qwen/Qwen2.5-1.5B-Instruct`

**For Production:**
- `Qwen/Qwen2.5-7B-Instruct` (recommended)
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

## Usage

### Command Line Interface

The modern CLI provides several commands:

```bash
# Run the web application
python main.py run --host 0.0.0.0 --port 7860

# Initialize database
python main.py init

# Check service health
python main.py health

# Show version
python main.py version

# Start all services (vLLM + app)
python scripts/start.py start

# Stop all services
python scripts/start.py stop

# Check service status
python scripts/start.py status

# View logs
python scripts/start.py logs
```

### Web Interface

After starting the application, access:
- **Web interface**: http://localhost:7860
- **vLLM API**: http://localhost:8000

### Programmatic Usage

```python
from citetube.ingestion.ingest import ingest_video
from citetube.retrieval.retrieve import hybrid_search
from citetube.llm.llm import answer_question

# Ingest a YouTube video
video_id, metadata = ingest_video("https://www.youtube.com/watch?v=VIDEO_ID")
print(f"Ingested video: {metadata['title']}")

# Search for relevant segments
segments = hybrid_search("What is machine learning?", video_id)

# Get answer with citations
response = answer_question("What is machine learning?", segments)
print(response['answer'])
```

## Development

### Available Commands

```bash
# Smart application management (recommended)
python main.py run          # Smart run - handles everything automatically
python main.py init         # Initialize database
python main.py health       # Check service health
python main.py version      # Show version

# Manual service management
python main.py start        # Start all services (vLLM + app)
python main.py stop         # Stop all services
python main.py status       # Check service status
python main.py logs         # View vLLM logs

# Testing
pytest tests/ -v           # Run tests
```

### Code Quality

The project uses modern Python tooling:

- **CLI**: Typer with Rich for beautiful output
- **Testing**: Pytest
- **Linting**: Ruff (available in virtual environment)

### Project Structure

```
CiteTube/
├── src/citetube/           # Main package
│   ├── core/              # Core functionality
│   │   ├── config.py      # Configuration management
│   │   ├── db.py          # Database operations
│   │   ├── models.py      # Data models
│   │   └── logging_config.py
│   ├── ingestion/         # Video ingestion
│   │   └── ingest.py      # Transcript fetching and processing
│   ├── retrieval/         # Search and retrieval
│   │   └── retrieve.py    # Hybrid search implementation
│   ├── llm/              # Language model interface
│   │   ├── llm.py        # Main LLM interface
│   │   └── vllm_client.py # vLLM client
│   └── ui/               # User interface
│       └── app.py        # Gradio web app
├── tests/                # Test files
├── docker/               # Docker configuration
├── data/                 # Data storage
│   └── logs/            # Application logs
├── main.py              # Smart CLI - handles everything!
├── pyproject.toml       # Modern Python configuration
├── requirements.txt     # Dependencies
└── .env.example         # Environment template
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific tests
pytest tests/test_vllm.py -v

# Run with coverage (if pytest-cov is installed)
pytest --cov=src/citetube tests/
```

## Performance

- **Embedding Generation**: ~100-500 tokens/second
- **Search**: Sub-second retrieval for most queries
- **Answer Generation**: 10-50 tokens/second (depends on model and hardware)

## Troubleshooting

### Common Issues

1. **Database Connection Failed**:
   ```bash
   # Check PostgreSQL status
   sudo systemctl status postgresql
   
   # Test connection
   python main.py health
   ```

2. **vLLM Server Not Starting**:
   ```bash
   # Check GPU availability
   nvidia-smi
   
   # View vLLM logs
   python scripts/start.py logs
   ```

3. **Out of Memory**:
   - Reduce `VLLM_GPU_MEMORY_UTILIZATION` in `.env`
   - Use a smaller model (e.g., `Qwen/Qwen2.5-0.5B-Instruct`)

### Getting Help

- Check the logs: `python scripts/start.py logs`
- Test services: `python main.py health`
- View service status: `python scripts/start.py status`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.