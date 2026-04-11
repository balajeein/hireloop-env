"""
Session Management
==================
UUID-based session isolation for concurrent access to the HireLoop environment.
Each session maintains its own independent HireLoopEnv instance.
Sessions expire after 1 hour to prevent memory leaks.
"""

import uuid
import time
from typing import Dict, Optional

from server.env import HireLoopEnv

_sessions: Dict[str, HireLoopEnv] = {}
_session_timestamps: Dict[str, float] = {}

SESSION_TTL = 3600  # 1 hour in seconds

# Default session slot for backward-compatible GET /reset
_DEFAULT_SESSION_ID = "__legacy__"


def _cleanup_expired():
    """Remove sessions older than SESSION_TTL."""
    now = time.time()
    expired = [
        sid for sid, ts in _session_timestamps.items()
        if now - ts > SESSION_TTL
    ]
    for sid in expired:
        _sessions.pop(sid, None)
        _session_timestamps.pop(sid, None)


def create_session() -> tuple:
    """
    Create a new session with an isolated environment instance.
    Cleans up expired sessions on every call.

    Returns:
        (session_id, env_instance)
    """
    _cleanup_expired()
    session_id = str(uuid.uuid4())
    _sessions[session_id] = HireLoopEnv()
    _session_timestamps[session_id] = time.time()
    return session_id, _sessions[session_id]


def get_session(session_id: str) -> Optional[HireLoopEnv]:
    """Retrieve an existing session's environment, or None if not found."""
    return _sessions.get(session_id)


def delete_session(session_id: str):
    """Remove a session and free its resources."""
    _sessions.pop(session_id, None)
    _session_timestamps.pop(session_id, None)


def get_or_create_legacy_session() -> tuple:
    """
    Get or create the legacy default session for backward-compatible
    GET /reset calls (no session_id required).

    Returns:
        (session_id, env_instance)
    """
    if _DEFAULT_SESSION_ID not in _sessions:
        _sessions[_DEFAULT_SESSION_ID] = HireLoopEnv()
    return _DEFAULT_SESSION_ID, _sessions[_DEFAULT_SESSION_ID]