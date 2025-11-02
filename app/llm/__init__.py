"""LLM helpers: grounded answer generation over retrieved context."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "DEFAULT_CHAT_MODEL",
    "GENERAL_ASSISTANT_PROMPT",
    "GROUNDING_SYSTEM_PROMPT",
    "GroundedAnswer",
    "UNKNOWN_PHRASE",
    "USEFUL_RETRIEVAL_MAX_L2",
    "SourceRef",
    "build_grounded_messages",
    "chunks_to_source_refs",
    "create_chat_llm",
    "format_context_for_prompt",
    "generate_general_answer",
    "generate_grounded_answer",
    "print_grounded_result",
    "retrieval_is_useful",
]

if TYPE_CHECKING:
    from app.llm.generator import (
        DEFAULT_CHAT_MODEL,
        GENERAL_ASSISTANT_PROMPT,
        GROUNDING_SYSTEM_PROMPT,
        GroundedAnswer,
        UNKNOWN_PHRASE,
        USEFUL_RETRIEVAL_MAX_L2,
        SourceRef,
        build_grounded_messages,
        chunks_to_source_refs,
        create_chat_llm,
        format_context_for_prompt,
        generate_general_answer,
        generate_grounded_answer,
        print_grounded_result,
        retrieval_is_useful,
    )


def __getattr__(name: str) -> Any:
    if name in __all__:
        mod = import_module("app.llm.generator")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
