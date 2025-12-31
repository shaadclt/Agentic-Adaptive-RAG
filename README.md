# 🤖 Agentic Adaptive RAG with LangGraph

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-enabled-green.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Agentic Adaptive RAG is a production-ready framework for building self-correcting, reasoning-based LLM systems that dynamically choose between retrieval, web search, and generation.**

> Build intelligent RAG systems that know when to retrieve documents, search the web, or generate responses directly

An advanced Retrieval-Augmented Generation (RAG) system that intelligently integrates dynamic query analysis with self-correcting mechanisms to optimize response accuracy. Unlike traditional RAG approaches, this system adapts its strategy based on query complexity and context.

## 🌟 Key Features

- **🧠 Intelligent Query Routing**: Automatically determines whether to use local documents, web search, or direct LLM generation
- **📊 Multi-Stage Quality Assurance**: Document relevance assessment, hallucination detection, and answer quality evaluation
- **🔄 Self-Correcting Mechanisms**: Automatically triggers additional retrieval or regeneration when quality thresholds aren't met
- **🌐 Hybrid Knowledge Sources**: Seamlessly combines local vector store with real-time web search
- **⚡ Production-Ready**: Built with LangGraph for robust state management and workflow orchestration

## 🏗️ System Architecture

The system implements three different retrieval strategies based on query complexity:

- **No Retrieval**: For queries answerable from parametric knowledge
- **Single-Step Retrieval**: For simple queries requiring document lookup
- **Multi-Hop Retrieval**: For complex queries requiring reasoning across multiple sources

<img width="2607" height="3575" alt="RAG Query Pipeline with-2025-12-31-083543" src="https://github.com/user-attachments/assets/937e752d-ce51-43e6-84a3-54eca371b6de" />


## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- UV package manager (recommended) or pip

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/Agentic-Adaptive-RAG.git
cd Agentic-Adaptive-RAG
```

2. **Set up virtual environment with UV**
```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
uv pip install -r requirements.txt
```

4. **Configure environment variables**
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=agentic-rag
```

### Getting Your API Keys

- **Google AI API Key**: Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
- **Tavily API Key**: Sign up at [Tavily](https://tavily.com/)
- **LangChain API Key**: Get it from [LangSmith](https://smith.langchain.com/)

## 🎯 Usage

### 1. Initialize the Vector Database

```bash
python ingestion.py
```

This creates a local Chroma vector store with documents about AI agents, prompt engineering, and adversarial attacks.

### 2. Run the Interactive Chatbot

```bash
python main.py
```

### 3. Example Interaction

```
🤖 Advanced RAG Chatbot
Welcome! Ask me anything or type 'quit', 'exit', or 'bye' to leave.

💬 You: what is agent memory?
🤔 Bot: Thinking...
---ROUTE QUESTION---
---ROUTE QUESTION TO RAG---
---RETRIEVE---
---CHECK DOCUMENT RELEVANCE TO QUESTION---
---GRADE: DOCUMENT RELEVANT---
---ASSESS GRADED DOCUMENTS---
---DECISION: GENERATE---
---GENERATE---
---CHECK HALLUCINATIONS---
---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---
---GRADE GENERATION vs QUESTION---
---DECISION: GENERATION ADDRESSES QUESTION---

🤖 Bot: Agent memory is a key component of AI systems that enables agents to store, retrieve, and utilize information across interactions...
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python -m pytest . -s -v
```

The test suite validates:
- Document relevance grading
- Hallucination detection
- Query routing logic
- Generation quality
- End-to-end workflow

## 📂 Project Structure

```
building-adaptive-rag/
├── graph/
│   ├── chains/                 # LLM processing chains
│   │   ├── tests/
│   │   │   ├── __init__.py
│   │   │   └── test_chains.py
│   │   ├── __init__.py
│   │   ├── answer_grader.py    # Answer quality evaluation
│   │   ├── generation.py       # Response generation
│   │   ├── hallucination_grader.py  # Hallucination detection
│   │   ├── retrieval_grader.py # Document relevance scoring
│   │   └── router.py           # Query routing logic
│   ├── nodes/                  # Workflow nodes
│   │   ├── __init__.py
│   │   ├── generate.py         # Generation node
│   │   ├── grade_documents.py  # Document grading node
│   │   ├── retrieve.py         # Retrieval node
│   │   └── web_search.py       # Web search node
│   ├── __init__.py
│   ├── consts.py              # System constants
│   ├── graph.py               # Main workflow orchestration
│   └── state.py               # State management
├── static/                     # Assets and diagrams
├── .env                       # Environment variables
├── .gitignore
├── ingestion.py               # Document ingestion pipeline
├── main.py                    # Application entry point
├── model.py                   # Model configurations
├── README.md
└── requirements.txt
```

## 🔧 Key Components

### State Management
The system uses a `GraphState` TypedDict that flows through all workflow nodes:
- `question`: User's input query
- `generation`: LLM's response
- `web_search`: Boolean flag for web search necessity
- `documents`: Retrieved documents from local and web sources

### Workflow Nodes

1. **Query Router**: Determines optimal information source (vectorstore vs. web search)
2. **Document Retriever**: Fetches relevant documents from local vector store
3. **Document Grader**: Evaluates document relevance and triggers web search if needed
4. **Web Search**: Queries external sources for additional information
5. **Generator**: Creates responses using retrieved context
6. **Quality Graders**: Assess hallucinations and answer relevance

### Decision Logic

The system implements intelligent decision-making at multiple points:
- Routes queries based on content domain
- Grades document relevance and triggers web search for insufficient results
- Detects hallucinations and regenerates responses when needed
- Evaluates answer quality and seeks additional information if required

## 🛠️ Configuration

### Model Configuration

Edit `model.py` to customize your language and embedding models:

```python
# Language Model Options
from langchain_aws import ChatBedrock
from langchain_google_genai import GoogleGenerativeAIEmbeddings

llm_model = ChatBedrock(model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0")
embed_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
```

### Document Sources

Customize the knowledge base by editing the URLs in `ingestion.py`:

```python
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
```

## 📈 Performance Optimization

- **Chunk Size**: Optimized at 250 tokens for better embedding quality
- **Retrieval Limit**: Configurable number of documents retrieved
- **Web Search Results**: Limited to 3 results for efficiency
- **Caching**: Persistent Chroma vector store for faster subsequent queries

## 🔬 Advanced Features

### Quality Assurance Pipeline

1. **Document Relevance Scoring**: Binary classification of document relevance
2. **Hallucination Detection**: Verification that responses are grounded in evidence
3. **Answer Quality Assessment**: Evaluation of response completeness and relevance

### Adaptive Routing

The system intelligently routes queries based on:
- Content domain analysis
- Query complexity assessment
- Available knowledge sources
- Previous retrieval success rates

## 🚧 Future Enhancements

- [ ] **LLM Fallback State**: Direct LLM responses for conversational queries
- [ ] **Enhanced Router**: Three-way routing (vectorstore/websearch/llm_fallback)
- [ ] **Multi-Modal Support**: Image and document understanding
- [ ] **Conversation Memory**: Context preservation across interactions
- [ ] **Custom Evaluation Metrics**: Domain-specific quality assessment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the foundational RAG framework
- [LangGraph](https://github.com/langchain-ai/langgraph) for stateful workflow orchestration
- [Mistral AI](https://mistral.ai/) for inspiration and research contributions
- Research paper: "Adaptive RAG" by Soyeong Jeong et al., 2024

## 📊 Citation

If you use this project in your research, please cite:

```bibtex
@software{agentic_adaptive_rag,
  title={Agentic Adaptive RAG with LangGraph},
  author={Mohamed Shaad},
  year={2025},
  url={https://github.com/shaadclt/Agentic-Adaptive-RAG}
}
```

## 📧 Contact

- **Author**: Mohamed Shaad
- **LinkedIn**: [Connect on LinkedIn](https://linkedin.com/in/mshaadk)

---

⭐ **Star this repository if you find it helpful!**
