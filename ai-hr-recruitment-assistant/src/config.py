"""
Central configuration for the AI HR Recruitment Assistant.
Loads settings from environment variables (.env file) so no secrets
ever live in source code.
"""

import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/knowledge_base/chroma_store")

UPLOAD_DIR = "uploads"
REPORTS_DIR = "reports"
KNOWLEDGE_BASE_DIR = "data/knowledge_base"


def require_api_key() -> None:
    """Raise a clear, friendly error if no API key has been configured."""
    if not ANTHROPIC_API_KEY:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Copy .env and add your "
            "Anthropic API key from https://console.anthropic.com/"
        )
