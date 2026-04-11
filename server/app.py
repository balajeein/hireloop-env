import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.env import HireLoopEnv
from models import HireLoopAction, HireLoopObservation

try:
    from openenv_core.env_server import create_fastapi_app
    env = HireLoopEnv()
    app = create_fastapi_app(env, HireLoopAction, HireLoopObservation)
except (ImportError, Exception):
    # Fallback to existing api.py if create_fastapi_app not available
    from api import app