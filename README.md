# ✈️ AI Marketing Agents for Aircraft Industry

An intelligent AI-powered system that combines CrewAI agents with advanced RAG (Retrieval-Augmented Generation) capabilities to provide comprehensive aircraft inventory management and marketing content generation.


## 🎥 Demo Videos

### Demo 1: Self Fine tuned LLM Expert
**Drive Link**: [AI Marketing Agent - Basic Demo](https://drive.google.com/file/d/your-demo-1-link/view)

**Features Demonstrated**:
- Aircraft inventory search
- Marketing content generation
- Basic query processing
- Interface navigation

### Demo 2: Multi-Agent Orchestration Framework
**Drive Link**: [AI Marketing Agent - Advanced Demo](https://drive.google.com/file/d/your-demo-2-link/view)

**Features Demonstrated**:
- CrewAI multi-agent workflow
- Hybrid search capabilities
- Real-time streaming responses
- Complex query routing



## 🚀 Features

### 🤖 Multi-Agent System
- **Router Agent**: Intelligently routes queries to specialized agents
- **Inventory Agent**: Aircraft search, specifications, and inventory management
- **Marketing Agent**: Content creation and marketing strategy
- **General Agent**: Friendly chat and guidance

### 🔍 Advanced Search Capabilities
- **Hybrid Search**: Combines FAISS vector search with BM25 text search
- **Dual-Mode RAG**: Separate indices for inventory and marketing content
- **Real-time Processing**: Instant responses with streaming capabilities

### 📊 Data Management
- **Aircraft Inventory**: 1000+ unique aircraft with detailed specifications
- **Marketing Content**: Pre-generated marketing posts and industry terms
- **Vector Embeddings**: Efficient semantic search using SentenceTransformers

### 🎯 Use Cases
- Aircraft specification queries and comparisons
- Marketing content generation for aviation industry
- Inventory management and search
- Customer support and guidance

## 📋 Prerequisites

- Python 3.8+
- OpenAI API Key
- AWS SageMaker (optional, for enhanced LLM capabilities)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AI_Marketing_Agent_Aircrafts
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

## 🚀 Quick Start

### Option 1: Streamlit Interface (Recommended)
```bash
streamlit run streamlit_interface.py
```

### Option 2: CrewAI Test Interface
```bash
streamlit run crew_ai_test.py
```

### Option 3: Pre-trained Model Interface
```bash
streamlit run streamlit_pretrained.py
```

## 📁 Project Structure

```
AI_Marketing_Agent_Aircrafts/
├── 📄 streamlit_interface.py      # Main Streamlit interface
├── 📄 crew_ai_test.py            # CrewAI multi-agent system
├── 📄 streamlit_pretrained.py    # Pre-trained model interface
├── 📄 RAG_generation.py          # RAG system implementation
├── 📄 requirements.txt           # Python dependencies
├── 📄 Working prompt.txt         # System prompts
├── 
├── 📊 Data Files
│   ├── marketing_metadata.pkl    # Marketing content metadata
│   ├── marketing_faiss.idx       # Marketing vector index
│   ├── inventory_metadata.pkl    # Aircraft inventory metadata
│   ├── inventory_faiss.idx       # Aircraft vector index
│   ├── inventory_bm25.pkl        # BM25 search index
│   └── raw_data/                 # Source CSV files
│
├── 🤖 Models
│   ├── fine_tuned_model/         # Fine-tuned language models
│   ├── llama-3-1-8b-instruct-*/  # Llama model variants
│   └── jumpstart-dft-llama-*/    # AWS JumpStart models
│
└── 📝 Training Data
    ├── train.jsonl               # Training dataset
    ├── train_fixed.jsonl         # Fixed training data
    └── train_final.jsonl         # Final training dataset
```



## 🔧 Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your-openai-api-key
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_DEFAULT_REGION=us-east-2
```

### Model Configuration
- **Default LLM**: OpenAI GPT models
- **Embedding Model**: all-MiniLM-L6-v2
- **Vector Index**: FAISS with L2 distance
- **Text Search**: BM25 algorithm

## 📊 Data Sources

### Aircraft Inventory
- **Source**: `raw_data/aircraft_inventory_1000_unique.csv`
- **Records**: 1000+ unique aircraft
- **Fields**: Manufacturer, model, specifications, pricing, features

### Marketing Content
- **Posts**: `raw_data/marketing_posts_1000.csv`
- **Terms**: `raw_data/marketing_terms_1000.csv`
- **Content**: Industry-specific marketing materials

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation in the code comments


---

**Built with ❤️ for the Aviation Industry** 