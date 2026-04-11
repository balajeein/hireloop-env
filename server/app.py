import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import HireLoopAction, HireLoopObservation

try:
    from openenv_core.env_server import create_fastapi_app
    from server.env import HireLoopEnv
    env = HireLoopEnv()
    app = create_fastapi_app(env, HireLoopAction, HireLoopObservation)
except (ImportError, Exception):
    from api import app

import uvicorn

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()