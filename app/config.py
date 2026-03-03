"""Load settings from environment variables; optional local files for dev only."""

import os

from app.env_loader import load_repo_dotenv

load_repo_dotenv()


def get_openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Set it in the environment (e.g. Render), "
            "or create a gitignored .env.local / .env from .env.example."
        )
    return key
