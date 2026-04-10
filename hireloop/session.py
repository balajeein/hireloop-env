"""
Session Management
==================
UUID-based session isolation for concurrent access to the HireLoop environment.
Each session maintains its own independent HireLoopEnv instance.
"""

import uuid
from typing import Dict, Optional

from hireloop.env import HireLoopEnv

_sessions: Dict[str, HireLoopEnv] = {}

# Default session slot for backward-compatible GET /reset
_DEFAULT_SESSION_ID = "__legacy__"


def create_session() -> tuple:
    """
    Create a new session with an isolated environment instance.

    Returns:
        (session_id, env_instance)
    """
    session_id = str(uuid.uuid4())
    _sessions[session_id] = HireLoopEnv()
    return session_id, _sessions[session_id]


def get_session(session_id: str) -> Optional[HireLoopEnv]:
    """Retrieve an existing session's environment, or None if not found."""
    return _sessions.get(session_id)


def delete_session(session_id: str):
    """Remove a session and free its resources."""
    _sessions.pop(session_id, None)


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
