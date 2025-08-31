# CiteTube

> **Ask questions about YouTube videos and get answers with exact timestamps!**

CiteTube is an AI-powered application that lets you chat with YouTube videos. Simply paste a YouTube link, ask questions, and receive answers with precise timestamps showing exactly when that information appears in the video.

## ✨ Features

- **🔒 Complete Privacy** - Everything runs locally on your machine
- **📍 Precise Timestamps** - Every answer includes exact video timestamps
- **🚀 Smart Search** - AI-powered semantic search with keyword matching
- **💰 Zero Cost** - No API fees or subscriptions required
- **🤖 Local AI** - Powered by vLLM running on your hardware

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **PostgreSQL**
- **8GB+ RAM** (more is better)
- **GPU** (optional, improves performance)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CiteTube
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the database**
   ```bash
   python main.py init
   ```

4. **Start CiteTube**
   ```bash
   python main.py run
   ```

5. **Open your browser**
   
   Navigate to: **http://localhost:7860**

That's it! The `run` command automatically handles configuration, starts the AI server, connects to the database, and launches the web interface.

## 🛠️ System Setup

### Ubuntu/Debian Setup

```bash
# Install system dependencies
sudo apt update
sudo apt install -y postgresql postgresql-contrib python3-pip python3-venv git

# Configure PostgreSQL
sudo systemctl start postgresql
sudo -u postgres psql -c "CREATE DATABASE citetube;"
sudo -u postgres psql -c "ALTER USER postgres PASSWORD '12345';"

# Install pgvector extension
cd /tmp
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
make && sudo make install
sudo -u postgres psql -d citetube -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Python Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

## ⚙️ Configuration

CiteTube works out of the box, but you can customize settings in the `.env` file:

```env
# AI Model Configuration
VLLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct

# Database Settings
DB_HOST=localhost
DB_PORT=5432
DB_NAME=citetube
DB_USER=postgres
DB_PASSWORD=12345

# Search Parameters
TOP_K=8
TEMPERATURE=0.1
```

### AI Model Options

| Model | Size | Speed | Quality | RAM Usage |
|-------|------|-------|---------|-----------|
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | ⚡⚡⚡ | ⭐⭐ | Low |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | ⚡⚡ | ⭐⭐⭐ | Medium |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | ⚡ | ⭐⭐⭐⭐ | High |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | ⚡ | ⭐⭐⭐⭐⭐ | High |

## 📖 Usage

1. **Start the application**
   ```bash
   python main.py run
   ```

2. **Add a YouTube video**
   - Paste any YouTube URL in the web interface
   - Wait for processing (first-time setup takes a few minutes)

3. **Ask questions**
   - "What is this video about?"
   - "When do they mention machine learning?"
   - "Summarize the key points"

4. **Get timestamped answers**
   - Answers include precise timestamps like "At 2:35, the speaker mentions..."
   - Click timestamps to jump directly to that moment

## 🔧 Commands

| Command | Description |
|---------|-------------|
| `python main.py run` | Start all services |
| `python main.py health` | Check system status |
| `python main.py status` | View running services |
| `python main.py stop` | Stop all services |
| `python main.py logs` | View application logs |

## 🏗️ Architecture

CiteTube consists of four main components:

- **🗄️ PostgreSQL Database** - Stores video transcripts with vector embeddings
- **🤖 vLLM Server** - Local AI model for question answering
- **🔍 Search Engine** - Hybrid semantic and keyword search
- **🌐 Web Interface** - User-friendly chat interface

## 🐛 Troubleshooting

### Database Connection Issues
```bash
# Ensure PostgreSQL is running
sudo systemctl start postgresql

# Test connection
python main.py health
```

### vLLM Server Issues
- **Insufficient RAM**: Use a smaller model (e.g., `Qwen/Qwen2.5-0.5B-Instruct`)
- **No GPU**: vLLM will automatically fall back to CPU mode
- **Out of memory**: Close other applications or restart your system

### Getting Help
```bash
# Check system status
python main.py health

# View detailed logs
python main.py logs
```

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

**Ready to start chatting with YouTube videos?** Follow the [Quick Start](#-quick-start) guide above! 🚀