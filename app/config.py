"""Load settings from environment variables; optional local files for dev only."""

import os

from app.env_loader import load_repo_dotenv
from app.env_loader import is_openai_key_placeholder

load_repo_dotenv()


def get_openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Set it in the environment (e.g. Render), "
            "or create gitignored `.env.local` from `.env.local.template` (never commit secrets)."
        )
    bad, why = is_openai_key_placeholder(key)
    if bad:
        raise ValueError(
            "OPENAI_API_KEY looks like a placeholder value. "
            "Replace it with a real key in gitignored `.env.local` (preferred) or your shell. "
            f"(detected: {why})"
        )
    return key
