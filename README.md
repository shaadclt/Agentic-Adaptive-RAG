# Agentic Adaptive RAG

> A production-oriented Agentic RAG system built with LangGraph that dynamically routes queries between local knowledge retrieval and web search, validates retrieval and generated answers, detects hallucinations, applies security controls, and provides evaluation and observability.


## Overview

**Agentic Adaptive RAG** is an end-to-end Retrieval-Augmented Generation system designed around an agentic workflow rather than a fixed retrieval pipeline.

Instead of always performing vector search, the system evaluates the user's question and dynamically determines the appropriate path:

- Retrieve information from uploaded documents
- Fall back to web search when local knowledge is insufficient
- Generate an answer from trusted context
- Evaluate whether retrieved documents are relevant
- Check whether the generated answer is grounded
- Verify whether the answer actually addresses the question
- Retry generation when quality gates fail
- Record evaluation and observability information for each run
- Apply security controls against prompt injection and sensitive information leakage
- Support human approval before web search when required

The project includes a **FastAPI backend**, **Next.js frontend**, **LangGraph agent workflow**, vector-based retrieval, LLM-based grading, security controls, evaluation tracking, observability, and Docker deployment.


## Key Features

### Agentic Adaptive Retrieval

The system does not blindly retrieve documents for every query.

It dynamically routes requests based on the question:

```text
                    User Question
                         │
                         ▼
                  Security Checks
                         │
                         ▼
                   Query Router
                    /         \
                   /           \
                  ▼             ▼
          Local Knowledge     Web Search
             Retrieval        (when needed)
                  │             │
                  ▼             ▼
           Retrieval Grader     │
                  │             │
          ┌───────┴───────┐     │
          │               │     │
       Relevant        Not Relevant
          │               │     │
          │               └─────┘
          │                   │
          └─────────┬─────────┘
                    ▼
                Generation
                    │
                    ▼
          Hallucination Grader
                    │
                    ▼
              Answer Grader
                    │
              ┌─────┴─────┐
              │           │
             Pass        Fail
              │           │
              ▼           ▼
           Response     Retry
                         │
                         └──────► Generation
```


## Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                         Next.js UI                           │
│                                                              │
│  Chat • Document Upload • Evaluation • Observability         │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                       FastAPI Backend                         │
│                                                              │
│  API • Uploads • Chat • Evaluation • Observability           │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                       Security Layer                          │
│                                                              │
│  Prompt Injection • Jailbreak Detection • Secret Redaction  │
│  Document Injection Detection • Output Protection           │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                        LangGraph                              │
│                                                              │
│  Router → Retrieve → Grade → Generate → Validate → Retry     │
│                 │                                            │
│                 └──────────────► Web Search                   │
└───────────────┬───────────────────────────────┬──────────────┘
                │                               │
                ▼                               ▼
       Vector Store / RAG                  Web Search
                │                               │
                └───────────────┬───────────────┘
                                ▼
                           LLM / Groq
                                │
                                ▼
                    Evaluation & Observability
```


# Core Workflow

## 1. Query Routing

The system first determines whether the question can be answered using the available local knowledge or requires external information.

```text
Question
   │
   ▼
Router
   │
   ├── Local knowledge
   │
   └── Web search
```

This prevents unnecessary web searches when relevant uploaded knowledge is available.


## 2. Local Document Retrieval

Users can provide their own documents rather than relying on hardcoded knowledge sources.

Supported document formats include:

- PDF
- DOCX
- TXT
- Markdown

Documents are processed and added to the application's vector-based retrieval layer.


## 3. Retrieval Grading

Retrieved documents are evaluated for relevance to the user's question.

```text
Question + Retrieved Document
            │
            ▼
      Retrieval Grader
            │
       ┌────┴────┐
       ▼         ▼
     Relevant   Irrelevant
       │         │
       ▼         ▼
   Generation  Web Search
```

This allows the system to recognize when vector retrieval does not provide useful evidence.


## 4. Web Search Fallback

When local retrieval is insufficient, the workflow can route the request to web search.

```text
Local Knowledge
      │
      ├── Sufficient ──► Generate
      │
      └── Insufficient
               │
               ▼
          Web Search
               │
               ▼
            Generate
```

The project also supports human approval around web-search execution.


## 5. Hallucination Detection

Generated answers are checked against their available context.

```text
Generated Answer
       │
       ▼
Hallucination Grader
       │
   ┌───┴───┐
   ▼       ▼
Grounded  Not Grounded
   │          │
   ▼          ▼
 Continue    Retry
```

This provides a quality-control layer between generation and the final response.


## 6. Answer Quality Validation

The system also evaluates whether the generated answer actually addresses the original question.

```text
Question + Answer
       │
       ▼
 Answer Grader
       │
   ┌───┴───┐
   ▼       ▼
   Yes      No
   │         │
   ▼         ▼
Response    Retry
```

Generation therefore has multiple quality gates rather than simply returning the first LLM response.


# Security

Security is integrated into the application rather than treated as an external concern.

Implemented controls include:

### Prompt Injection Detection

Detects malicious instructions attempting to manipulate the agent.

### System Prompt Extraction Protection

Detects attempts to retrieve hidden system instructions.

### Jailbreak Detection

Identifies common attempts to bypass model or application restrictions.

### Document Injection Detection

Uploaded or retrieved content is treated as untrusted information rather than trusted instructions.

### Untrusted Content Handling

External content is wrapped and handled separately from trusted application instructions.

### Output Protection

The output layer can redact sensitive information such as:

- API keys
- Groq secrets
- Environment secrets

### Output Length Protection

Responses are constrained to prevent uncontrolled output.


# Evaluation

The project includes automated evaluation components for the RAG and agent workflow.

Evaluation covers:

- Retrieval quality
- Hallucination/grounding
- Answer relevance
- Routing behavior
- Generation retry behavior
- Response validation
- Evaluation tracking

Evaluation results can be persisted for later analysis.


# Observability

The application includes an observability layer for tracking agent executions.

Recorded information includes:

- Selected route
- Number of retrieved documents
- Grounding results
- Retry count
- Latency
- Source counts
- Human-in-the-loop usage
- Evaluation information

This makes it possible to inspect how the agent arrived at a response rather than treating the LLM as a black box.


# Human-in-the-Loop

The architecture includes a human approval step for web-search execution.

```text
Agent decides Web Search
          │
          ▼
     Approval Step
       /       \
      /         \
 Approve       Reject
    │             │
    ▼             ▼
Web Search     Rejection
```

This provides an additional control point before external retrieval is executed.


# Technology Stack

## Backend

- Python
- FastAPI
- LangGraph
- LangChain
- Groq
- Pydantic

## AI / ML

- Retrieval-Augmented Generation
- Agentic workflows
- LLM-based grading
- Hallucination detection
- Prompt engineering
- Embeddings
- Vector search

## Storage

- Chroma
- Document storage
- Evaluation history
- Observability history

## Frontend

- Next.js
- React
- TypeScript

## Deployment

- Docker
- Docker Compose
- Uvicorn

## Testing

- Pytest


# Project Structure

```text
Agentic-Adaptive-RAG/
│
├── backend/
│   ├── api.py
│   ├── config.py
│   ├── model.py
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── response.py
│   ├── sources.py
│   ├── evaluation.py
│   ├── observability.py
│   ├── main.py
│   │
│   ├── graph/
│   │   ├── graph.py
│   │   ├── state.py
│   │   ├── consts.py
│   │   ├── context.py
│   │   │
│   │   ├── chains/
│   │   │   ├── answer_grader.py
│   │   │   ├── generation.py
│   │   │   ├── hallucination_grader.py
│   │   │   ├── retrieval_grader.py
│   │   │   └── router.py
│   │   │
│   │   └── nodes/
│   │       ├── build_response.py
│   │       ├── generate.py
│   │       ├── grade_documents.py
│   │       ├── increment_retry.py
│   │       ├── retrieve.py
│   │       ├── web_search.py
│   │       ├── web_search_approval.py
│   │       └── web_search_rejected.py
│   │
│   └── security/
│       ├── content_guard.py
│       ├── output_guard.py
│       ├── prompt_guard.py
│       ├── security_config.py
│       └── tests/
│
├── evaluation/
│   ├── benchmarks/
│   ├── results/
│   └── ragas/
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── public/
│   ├── Dockerfile
│   └── next.config.ts
│
├── data/
├── uploads/
│
├── .chroma/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```


# Installation

## Prerequisites

Make sure you have:

- Python 3.11+
- Node.js 20+
- Docker Desktop
- Git
- A Groq API key


# Environment Variables

Create a `.env` file in the project root.

Example:

```env
GROQ_API_KEY=your_groq_api_key

MAX_UPLOAD_SIZE_MB=10

FRONTEND_URL=http://localhost:3000
```

Use `.env.example` as the template for the complete configuration.

**Never commit your real API key to Git.**


# Running Locally

## 1. Clone the repository

```powershell
git clone <your-repository-url>
cd Agentic-Adaptive-RAG
```

## 2. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate
```

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## 4. Configure environment variables

Create `.env` and add your API configuration.

## 5. Start the backend

```powershell
uvicorn backend.api:api --reload
```

Backend:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

Health check:

```text
http://localhost:8000/health
```

## 6. Start the frontend

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Frontend:

```text
http://localhost:3000
```


# Running with Docker

Docker Compose is provided for running the complete application.

Make sure **Docker Desktop is running** before executing the commands below.

From the project root:

```powershell
docker compose build
```

Start the application:

```powershell
docker compose up
```

Or run in detached mode:

```powershell
docker compose up -d
```

The services are exposed at:

```text
Frontend:
http://localhost:3000

Backend:
http://localhost:8000

API documentation:
http://localhost:8000/docs
```

Check container status:

```powershell
docker compose ps
```

View logs:

```powershell
docker compose logs -f
```

Stop the application:

```powershell
docker compose down
```


# API

The FastAPI backend exposes endpoints for:

- Health checks
- Document upload
- Document management
- Chat/query execution
- Evaluation
- Observability

Interactive API documentation is available through:

```text
http://localhost:8000/docs
```


# Testing

The project includes automated tests covering the core agent, evaluation, response, and security components.

Run the complete test suite:

```powershell
python -m pytest -v
```

Current verification:

```text
33 passed
```

The test suite covers:

```text
✓ Retrieval grading
✓ Hallucination grading
✓ Answer grading
✓ Query routing
✓ Local RAG path
✓ Web-search fallback
✓ Generation retries
✓ Quality gates
✓ Evaluation tracking
✓ Response validation
✓ Prompt injection protection
✓ System prompt extraction protection
✓ Jailbreak detection
✓ Document injection detection
✓ Untrusted content handling
✓ Secret redaction
✓ Output protection
```


# Example Workflow

### Step 1 — Upload a document

Upload a supported document through the application.

```text
PDF / DOCX / TXT / MD
          │
          ▼
      Ingestion
          │
          ▼
      Chunking
          │
          ▼
   Vector Storage
```

### Step 2 — Ask a question

```text
"What are the main conclusions of this document?"
```

### Step 3 — Agent processes the question

```text
Question
   ↓
Security Check
   ↓
Route
   ↓
Retrieve
   ↓
Grade Documents
   ↓
Generate
   ↓
Check Grounding
   ↓
Check Answer Quality
   ↓
Final Response
```

### Step 4 — Inspect execution

Use the observability interface to inspect the execution path and evaluation information.


# Why This Project Is Agentic

A conventional RAG pipeline might look like:

```text
Question
   ↓
Vector Search
   ↓
LLM
   ↓
Answer
```

This project introduces decision-making and feedback loops:

```text
                         ┌───────────────┐
                         │    Router     │
                         └───────┬───────┘
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
             Local Retrieval              Web Search
                   │
                   ▼
           Retrieval Grader
                   │
            ┌──────┴──────┐
            ▼             ▼
         Relevant      Not Relevant
            │             │
            └──────┬──────┘
                   ▼
               Generation
                   │
                   ▼
          Hallucination Check
                   │
                   ▼
             Answer Check
                   │
            ┌──────┴──────┐
            ▼             ▼
          Accept         Retry
                            │
                            └──────► Generation
```

The system therefore combines:

- Routing
- Tool selection
- Retrieval
- Evaluation
- Feedback
- Retry
- Human approval

within a stateful LangGraph workflow.


# Engineering Highlights

### Adaptive Retrieval

Rather than assuming vector search is always sufficient, the system evaluates retrieval quality and can transition to external search.

### Self-Evaluation

The system evaluates both retrieved evidence and generated responses.

### Quality-Controlled Generation

Generation is not automatically accepted. Hallucination and answer-quality checks determine whether the response should be returned or regenerated.

### Security-Aware RAG

Documents and external content are treated as potentially untrusted input, with dedicated controls for prompt injection and output leakage.

### Observability

Agent execution information is recorded so the behavior of the system can be analyzed.

### Containerized Deployment

Both frontend and backend are containerized and orchestrated using Docker Compose.


# Testing Philosophy

The project tests the behavior of individual components as well as the agent workflow.

The goal is not simply to verify that functions execute successfully, but to validate important agent behaviors:

```text
Routing
Retrieval
Fallback
Generation
Validation
Retry
Security
Evaluation
```


# Future Improvements

The current implementation focuses on the core Agentic RAG architecture.

Potential future production improvements include:

- Authentication and authorization
- API rate limiting
- Advanced document malware scanning
- More comprehensive file-type validation
- ZIP/archive resource limits
- Dependency vulnerability scanning
- Container image scanning
- Distributed observability
- Production vector database
- Background document processing
- Streaming responses
- Advanced RAG evaluation dashboards
- Persistent conversation memory
- Multi-user support
- Cloud deployment

These are intentionally treated as future production-hardening improvements rather than prerequisites for the current system.


# Project Status

| Component | Status |
|---|---|
| Agentic workflow | ✅ |
| LangGraph orchestration | ✅ |
| Adaptive routing | ✅ |
| Vector retrieval | ✅ |
| Web-search fallback | ✅ |
| Retrieval grading | ✅ |
| Hallucination grading | ✅ |
| Answer grading | ✅ |
| Generation retry | ✅ |
| Human-in-the-loop | ✅ |
| Security layer | ✅ |
| Evaluation | ✅ |
| Observability | ✅ |
| FastAPI backend | ✅ |
| Next.js frontend | ✅ |
| Docker backend | ✅ |
| Docker frontend | ✅ |
| Automated tests | ✅ 33 passed |



# License

This project is intended as a portfolio and learning project.

# Author

**Mohamed Shaad**

Machine Learning Engineer | Generative AI | Agentic AI | LLM Systems

```text
Python • LangGraph • LangChain • RAG • LLMs
FastAPI • Docker • Vector Search • Generative AI
```
