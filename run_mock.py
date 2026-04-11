import requests
from hireloop.env import HireLoopEnv
from api import _run_heuristic_task

env = HireLoopEnv()
print("Run 1")
print(_run_heuristic_task(env, "resume"))
print(_run_heuristic_task(env, "offer"))
print(_run_heuristic_task(env, "communication"))
