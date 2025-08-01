"""LLM helpers: grounded answer generation over retrieved context."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "DEFAULT_CHAT_MODEL",
    "GROUNDING_SYSTEM_PROMPT",
    "GroundedAnswer",
    "UNKNOWN_PHRASE",
    "SourceRef",
    "build_grounded_messages",
    "chunks_to_source_refs",
    "create_chat_llm",
    "format_context_for_prompt",
    "generate_grounded_answer",
    "print_grounded_result",
]

if TYPE_CHECKING:
    from app.llm.generator import (
        DEFAULT_CHAT_MODEL,
        GROUNDING_SYSTEM_PROMPT,
        GroundedAnswer,
        UNKNOWN_PHRASE,
        SourceRef,
        build_grounded_messages,
        chunks_to_source_refs,
        create_chat_llm,
        format_context_for_prompt,
        generate_grounded_answer,
        print_grounded_result,
    )


def __getattr__(name: str) -> Any:
    if name in __all__:
        mod = import_module("app.llm.generator")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
