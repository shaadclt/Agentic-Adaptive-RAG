import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


# -----------------------------
# LLM Configuration
# -----------------------------

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "openai/gpt-oss-120b",
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
# Groq LLM
# -----------------------------

llm_model = ChatGroq(
    model=LLM_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=LLM_TEMPERATURE,
)


# -----------------------------
# Google Gemini Embeddings
# -----------------------------

embed_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
)