"""Load settings from environment variables (via .env for local dev)."""

import os

from dotenv import load_dotenv

load_dotenv()


def get_openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Copy .env.example to .env and set your key."
        )
    return key

# Modified 2024-08-28

# Modified 2024-09-19

# Modified 2024-10-13

# Modified 2024-08-01

# Modified 2024-08-28

# Modified 2024-09-19

# Modified 2025-08-28

# Modified 2025-09-19

# Modified 2025-10-13

# Modified 2025-08-01
