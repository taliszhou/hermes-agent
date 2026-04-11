"""Global request throttle for local LLM inference servers.

When using a local model (ollama, vllm, lmstudio, etc.), the inference server
typically handles only 1-2 concurrent requests efficiently.  This module
provides a process-wide semaphore that all LLM call sites acquire before
sending a request, preventing request pile-up on slow local backends.

Cloud API paths are unaffected — the semaphore is only activated when local
inference mode is detected or explicitly enabled.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Module state (configured once at startup via ``configure()``)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_configured = False
_local_mode = False
_config: Dict[str, Any] = {}
_semaphore: threading.Semaphore | None = None

# Provider aliases that imply local inference.
_LOCAL_PROVIDERS = frozenset({
    "lmstudio", "lm-studio", "lm_studio",
    "vllm",
    "llamacpp", "llama.cpp", "llama-cpp",
    "local",
})


def configure() -> None:
    """Read config and initialise the throttle.  Idempotent — safe to call
    from every ``AIAgent.__init__`` including subagents."""
    global _configured
    with _lock:
        if _configured:
            return
        _apply_config()
        _configured = True


def reconfigure(*, base_url: str = "", provider: str = "") -> None:
    """Re-evaluate local mode after a runtime model switch (``/model``).

    Called from ``AIAgent.switch_model()`` with the new endpoint details.
    If ``local_inference.enabled`` is explicitly ``true`` or ``false`` in
    config, the explicit setting is honored and the runtime hints are
    ignored.  Only in ``"auto"`` mode do *base_url* and *provider* matter.
    """
    global _configured
    with _lock:
        _apply_config(runtime_base_url=base_url, runtime_provider=provider)
        _configured = True


def _apply_config(
    *,
    runtime_base_url: str = "",
    runtime_provider: str = "",
) -> None:
    """Shared implementation for ``configure`` and ``reconfigure``."""
    global _local_mode, _config, _semaphore

    from hermes_cli.config import load_config

    full_cfg = load_config()
    li_cfg = full_cfg.get("local_inference", {})
    _config = dict(li_cfg)

    enabled = li_cfg.get("enabled", "auto")
    if isinstance(enabled, bool):
        _local_mode = enabled
    elif str(enabled).lower() == "true":
        _local_mode = True
    elif str(enabled).lower() == "false":
        _local_mode = False
    else:
        # Auto-detect from config first, then runtime hints
        _local_mode = _detect_local(full_cfg, runtime_base_url, runtime_provider)

    if _local_mode:
        max_concurrent = int(li_cfg.get("max_concurrent_requests", 1))
        _semaphore = threading.Semaphore(max(1, max_concurrent))
    else:
        _semaphore = None


def _detect_local(
    full_cfg: Dict[str, Any],
    runtime_base_url: str = "",
    runtime_provider: str = "",
) -> bool:
    """Heuristic: is the current provider a local inference server?"""
    # Explicit env-var override (supports both ON and OFF)
    env_val = os.getenv("HERMES_LOCAL_MODE", "").lower()
    if env_val in ("1", "true", "yes"):
        return True
    if env_val in ("0", "false", "no"):
        return False

    # Check runtime hints first (from /model switch)
    if runtime_base_url:
        url_lower = runtime_base_url.lower()
        if "localhost" in url_lower or "127.0.0.1" in url_lower:
            return True
    if runtime_provider and runtime_provider.lower() in _LOCAL_PROVIDERS:
        return True

    # Fall back to config.yaml model section
    model_cfg = full_cfg.get("model", {})
    if isinstance(model_cfg, dict):
        base_url = (model_cfg.get("base_url") or "").lower()
        if "localhost" in base_url or "127.0.0.1" in base_url:
            return True
        provider = (model_cfg.get("provider") or "").lower()
        if provider in _LOCAL_PROVIDERS:
            return True

    return False


# ---------------------------------------------------------------------------
# Public helpers — called from LLM call sites
# ---------------------------------------------------------------------------

@contextmanager
def throttle():
    """Context manager that acquires the global semaphore when local mode is
    active.  No-op when using cloud APIs."""
    if _semaphore is not None:
        _semaphore.acquire()
        try:
            yield
        finally:
            _semaphore.release()
    else:
        yield


def is_local_mode() -> bool:
    return _local_mode


def get_config() -> Dict[str, Any]:
    return _config


def get_timeout(base: float) -> float:
    """Apply the configured timeout multiplier when in local mode."""
    if not _local_mode:
        return base
    multiplier = float(_config.get("timeout_multiplier", 3.0))
    return base * multiplier


def get_tool_workers() -> int:
    """Max concurrent tool-execution threads (default 8, reduced in local mode)."""
    if _local_mode:
        return int(_config.get("max_tool_workers", 2))
    return 8  # _MAX_TOOL_WORKERS original default


def get_stream_retries() -> int:
    """Max stream retry attempts (default 2, reduced in local mode)."""
    if _local_mode:
        return int(_config.get("stream_retries", 0))
    return int(os.getenv("HERMES_STREAM_RETRIES", 2))
