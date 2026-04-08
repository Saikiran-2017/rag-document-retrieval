"""Placeholder detection for local secret templates (no real keys)."""

from __future__ import annotations

import os

from app.env_loader import is_openai_key_placeholder, load_repo_dotenv


def test_placeholder_templates_rejected():
    bad, _ = is_openai_key_placeholder("your_openai_api_key_here")
    assert bad
    bad, _ = is_openai_key_placeholder("sk-your-openai-api-key-here")
    assert bad


def test_real_key_shape_not_template():
    ok, _ = is_openai_key_placeholder("sk-proj-" + "a" * 40)
    assert not ok


def test_local_openai_key_overrides_placeholder_in_process(tmp_path, monkeypatch):
    """Shell/IDE may pre-set the sample key; .env.local must still win after merge."""
    real = "sk-proj-" + "a" * 40
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=sk-your-openai-api-key-here\n", encoding="utf-8"
    )
    (tmp_path / ".env.local").write_text(f"OPENAI_API_KEY={real}\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-your-openai-api-key-here")
    load_repo_dotenv(tmp_path)
    assert os.environ["OPENAI_API_KEY"] == real


def test_local_openai_key_overrides_placeholder_in_env_file_only(tmp_path, monkeypatch):
    """Real key in .env.local must win when .env holds only a sample key (empty process)."""
    real = "sk-proj-" + "b" * 40
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=sk-your-openai-api-key-here\n", encoding="utf-8"
    )
    (tmp_path / ".env.local").write_text(f"OPENAI_API_KEY={real}\n", encoding="utf-8")
    load_repo_dotenv(tmp_path)
    assert os.environ["OPENAI_API_KEY"] == real


def test_valid_process_openai_key_beats_env_files(tmp_path, monkeypatch):
    """Production-style process key must not be overridden by local dev files."""
    proc = "sk-proj-" + "c" * 40
    monkeypatch.setenv("OPENAI_API_KEY", proc)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-your-openai-api-key-here\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text("OPENAI_API_KEY=sk-your-openai-api-key-here\n", encoding="utf-8")
    load_repo_dotenv(tmp_path)
    assert os.environ["OPENAI_API_KEY"] == proc
