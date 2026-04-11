"""
HireLoop Environment Package
=============================
A multi-step reinforcement learning environment simulating a real-world
hiring pipeline across 3 escalating tasks: resume screening, offer
decision, and communication drafting.

Built on openenv-core framework.
"""

from hireloop.env import HireLoopEnv
from hireloop.utils.skills import check_negotiation_eligibility

__all__ = ["HireLoopEnv", "check_negotiation_eligibility"]
__version__ = "1.0.0"
