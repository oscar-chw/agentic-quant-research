"""Keep test-only broker receipts out of the developer's real user profile."""

import pytest

from qrae import codex_broker


@pytest.fixture(autouse=True)
def isolated_default_receipt_keys(monkeypatch, tmp_path_factory):
    # A sibling of test workspaces preserves the real outside-workspace check.
    # Explicit receipt_key_root arguments and their security tests are unchanged.
    receipt_root = tmp_path_factory.mktemp("test-receipt-store")
    monkeypatch.setattr(codex_broker, "_default_receipt_key_root", lambda: receipt_root)
