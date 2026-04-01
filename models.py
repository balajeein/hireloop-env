from pydantic import BaseModel
from typing import List, Optional, Dict


class Candidate(BaseModel):
    id: str
    name: str
    skills: List[str]
    years_experience: int
    expected_salary: int
    gender: Optional[str] = None
    nationality: Optional[str] = None


class JobDescription(BaseModel):
    role: str
    required_skills: List[str]
    max_salary: int
    seniority: str  # "junior" / "mid" / "senior"


class HireLoopState(BaseModel):
    job_description: JobDescription
    candidates: List[Candidate]
    shortlisted: List[str]
    rejected: List[str]
    step_count: int
    task_type: str = "resume"
    budget: Optional[int] = None
    offers_made: Optional[List[Dict]] = None
    emails_sent: Optional[List[Dict]] = None
    counterfactual: Optional[Dict] = None

class Action(BaseModel):
    type: str  # "accept" | "reject" | "offer" | "negotiate" | "write_email"
    candidate_id: str
    content: Optional[str] = None  # only required for write_email


class Reward(BaseModel):
    value: float
    shaped: bool = True
    min: float = -1.0
    max: float = 1.0