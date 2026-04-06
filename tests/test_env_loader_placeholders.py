"""Placeholder detection for local secret templates (no real keys)."""

from __future__ import annotations

from app.env_loader import is_openai_key_placeholder


def test_placeholder_templates_rejected():
    bad, _ = is_openai_key_placeholder("your_openai_api_key_here")
    assert bad
    bad, _ = is_openai_key_placeholder("sk-your-openai-api-key-here")
    assert bad


def test_real_key_shape_not_template():
    ok, _ = is_openai_key_placeholder("sk-proj-" + "a" * 40)
    assert not ok
