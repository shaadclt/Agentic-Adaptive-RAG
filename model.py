```python
import os

from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

load_dotenv()


# -----------------------------
# LLM Configuration
# -----------------------------

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gemini-1.5-flash",
)

LLM_TEMPERATURE = float(
    os.getenv(
        "LLM_TEMPERATURE",
        "0.7",
    )
)


# -----------------------------
# Embedding Configuration
# -----------------------------

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "gemini-embedding-001",
)


# -----------------------------
# Google Gemini LLM
# -----------------------------

llm_model = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=LLM_TEMPERATURE,
)


# -----------------------------
# Google Gemini Embeddings
# -----------------------------

embed_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
)
