"""T2.2: cloud sandbox mode (E2B) — graceful-no-key behavior.

We deliberately do NOT exercise an actual E2B call here. The point of
these tests is to verify that:

  1. The CLOUD enum value exists and dispatches.
  2. Without E2B_API_KEY, the executor returns a clear error result
     instead of crashing — so callers can fall back to restricted.
  3. The error message names the env var so users know how to enable it.
"""

from __future__ import annotations

from core.sandbox import ExecutionMethod, execute_code_safely


class TestCloudEnum:
    def test_cloud_value_resolves(self):
        assert ExecutionMethod("cloud") is ExecutionMethod.CLOUD


class TestCloudUnconfigured:
    def test_without_key_returns_unconfigured_result(self, monkeypatch):
        monkeypatch.delenv("E2B_API_KEY", raising=False)
        result = execute_code_safely("print('ok')", method="cloud")
        assert result["success"] is False
        assert result["method_used"] == "cloud_unconfigured"
        assert "E2B_API_KEY" in result["error"]

    def test_error_message_points_at_e2b(self, monkeypatch):
        monkeypatch.delenv("E2B_API_KEY", raising=False)
        result = execute_code_safely("x = 1", method="cloud")
        assert "e2b.dev" in result["error"].lower() or "E2B" in result["error"]
