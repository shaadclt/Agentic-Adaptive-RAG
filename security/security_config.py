"""
Central security configuration for the Agentic Adaptive RAG system.
"""

MAX_QUESTION_LENGTH = 4000

MAX_DOCUMENT_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

MAX_FILES_PER_UPLOAD = 5

MAX_OUTPUT_LENGTH = 12000


# Direct prompt-injection patterns.
#
# These are intentionally conservative. The goal is not to claim
# perfect detection, but to identify common high-confidence attacks
# before they reach the agent.
PROMPT_INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+)?previous\s+instructions\b",
    r"\bignore\s+(all\s+)?prior\s+instructions\b",
    r"\bdisregard\s+(all\s+)?previous\s+instructions\b",
    r"\bdisregard\s+(all\s+)?prior\s+instructions\b",
    r"\boverride\s+(all\s+)?previous\s+instructions\b",
    r"\bforget\s+(all\s+)?previous\s+instructions\b",
    r"\bdo\s+not\s+follow\s+(the\s+)?previous\s+instructions\b",
    r"\breveal\s+(your\s+)?system\s+prompt\b",
    r"\bshow\s+(me\s+)?(your\s+)?system\s+prompt\b",
    r"\bprint\s+(your\s+)?system\s+prompt\b",
    r"\brepeat\s+(your\s+)?system\s+prompt\b",
    r"\breveal\s+(your\s+)?hidden\s+instructions\b",
    r"\breveal\s+(your\s+)?developer\s+instructions\b",
    r"\bshow\s+(me\s+)?your\s+developer\s+message\b",
    r"\bignore\s+the\s+safety\s+rules\b",
    r"\bbypass\s+(the\s+)?safety\s+rules\b",
    r"\bbypass\s+(all\s+)?restrictions\b",
    r"\bjailbreak\b",
]


# Patterns indicating that retrieved content may itself contain
# instructions intended to manipulate the LLM.
CONTENT_INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+)?previous\s+instructions\b",
    r"\bignore\s+(all\s+)?prior\s+instructions\b",
    r"\bdisregard\s+(all\s+)?instructions\b",
    r"\boverride\s+(all\s+)?instructions\b",
    r"\breveal\s+(the\s+)?system\s+prompt\b",
    r"\breveal\s+(hidden|secret)\s+instructions\b",
    r"\bact\s+as\s+(a\s+)?system\b",
    r"\byou\s+are\s+now\s+the\s+system\b",
    r"\bdo\s+not\s+answer\s+the\s+user\b",
    r"\bexecute\s+the\s+following\s+instructions\b",
]


# Secrets that should never be returned to the user.
SECRET_PATTERNS = [
    r"gsk_[A-Za-z0-9_-]{20,}",
    r"tvly-[A-Za-z0-9_-]{20,}",
    r"AIza[A-Za-z0-9_-]{20,}",
    r"sk-[A-Za-z0-9_-]{20,}",
    r"Bearer\s+[A-Za-z0-9._~+/=-]{20,}",
    r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----",
]


SENSITIVE_ENVIRONMENT_VARIABLES = [
    "GROQ_API_KEY",
    "TAVILY_API_KEY",
    "GOOGLE_API_KEY",
    "LANGSMITH_API_KEY",
    "OPENAI_API_KEY",
    "DATABASE_URL",
]