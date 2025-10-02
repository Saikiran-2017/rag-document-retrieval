"""Re-exports ranked web search pipeline."""

from app.services.web_search_service import (
    WebSnippet,
    prepare_web_for_generation,
    search_web_for_chat,
    web_results_strong_enough,
    web_search_enabled,
)

__all__ = [
    "WebSnippet",
    "prepare_web_for_generation",
    "search_web_for_chat",
    "web_results_strong_enough",
    "web_search_enabled",
]
