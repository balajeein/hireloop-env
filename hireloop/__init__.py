"""
HireLoop Environment Package
=============================
A multi-step reinforcement learning environment simulating a real-world
hiring pipeline across 3 escalating tasks: resume screening, offer
decision, and communication drafting.

Built on openenv-core framework.
"""

from hireloop.env import HireLoopEnv

__all__ = ["HireLoopEnv"]
__version__ = "1.0.0"
