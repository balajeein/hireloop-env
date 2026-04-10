"""
HireLoop Task Modules
=====================
Each module implements reset, step, and score for one task type.
"""

from hireloop.tasks import resume, offer, communication

__all__ = ["resume", "offer", "communication"]
