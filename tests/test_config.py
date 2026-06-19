"""config.py の認証情報ヘルパの単体テスト（anthropic_api_key）。

- 未設定なら None。
- 設定されていれば値を返す。
- 空文字は None（`or None` の挙動）。
"""

from __future__ import annotations

from flowsignal import config


def test_anthropic_api_key_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert config.anthropic_api_key() is None


def test_anthropic_api_key_returns_value_when_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert config.anthropic_api_key() == "sk-ant-test"


def test_anthropic_api_key_blank_is_none(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    assert config.anthropic_api_key() is None
