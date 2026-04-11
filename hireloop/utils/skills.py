"""
Skill Utilities
================
Skill category matching, similarity checks, and negotiation eligibility logic.
Extracted from hireloop/tasks/offer.py — no logic changes.
"""

from typing import Optional, List


# ---------------------------------------------------------------------------
# Skill similarity categories — two skills are similar if same category
# ---------------------------------------------------------------------------
SKILL_CATEGORIES = {
    "frontend": ["react", "vue", "angular", "svelte", "javascript", "typescript", "html", "css", "redux", "node", "nodejs"],
    "backend": ["express", "django", "flask", "fastapi", "spring", "rails", "laravel", "graphql"],
    "mobile": ["swift", "ios", "android", "kotlin", "flutter", "react_native", "objective_c", "xcode", "swiftui"],
    "ml_ai": ["ml", "tensorflow", "pytorch", "keras", "scikit", "nlp", "bert", "transformers", "spacy", "deep_learning", "llm", "statistics"],
    "data": ["sql", "postgres", "mysql", "spark", "hadoop", "kafka", "airflow", "dbt", "mongodb", "redis"],
    "devops": ["docker", "kubernetes", "terraform", "ansible", "jenkins", "ci_cd", "helm", "linux", "bash"],
    "cloud": ["aws", "gcp", "azure", "lambda", "s3", "cloudformation", "cloud"],
    "systems": ["c++", "c", "rust", "go", "java", "scala", "kotlin", "cpp"],
    "scripting": ["python", "bash", "shell", "ruby", "perl"],
    "analytics": ["powerbi", "tableau", "excel", "looker", "statistics", "r", "siem"],
    "security": ["security", "networking", "penetration_testing", "firewall", "cisco"],
    "management": ["product_management", "agile", "scrum", "jira", "roadmapping", "data_analysis", "analytics"],
}


def get_skill_category(skill: str) -> Optional[str]:
    """Returns the category of a skill, or None if not found."""
    skill_lower = skill.lower()
    for category, skills in SKILL_CATEGORIES.items():
        if skill_lower in skills:
            return category
    return None


def are_skills_similar(skill_a: str, skill_b: str) -> bool:
    """Returns True if two skills belong to the same category."""
    cat_a = get_skill_category(skill_a)
    cat_b = get_skill_category(skill_b)
    if cat_a is None or cat_b is None:
        return False
    return cat_a == cat_b


def check_negotiation_eligibility(
    candidate_skills: list,
    required_skills: list
) -> dict:
    # Checks if a candidate is eligible for salary negotiation.

    # Rules:
    #  Zero exact matches → direct reject
    # All exact matches  → full offer, no negotiation
    # Exactly 1+ exact match AND remaining required skills
    # are similar (same category) → negotiate at 10% less
    # 1+ exact match BUT remaining are NOT similar → reject

    # Returns dict with eligible, reason, exact_matches, similar_matches


    candidate_set = set(s.lower() for s in candidate_skills)
    required_set = set(s.lower() for s in required_skills)

    # Find exact matches
    exact_matches = candidate_set & required_set
    missing_required = required_set - exact_matches

    # Rule 1: Zero exact matches → reject immediately
    if len(exact_matches) == 0:
        return {
            "eligible": False,
            "negotiable": False,
            "reason": "No exact skill matches. Direct reject.",
            "exact_matches": list(exact_matches),
            "similar_matches": [],
            "discount": 0.0,
        }

    # Rule 2: All required skills matched → full offer
    if len(missing_required) == 0:
        return {
            "eligible": True,
            "negotiable": False,
            "reason": "Full skill match. Standard offer.",
            "exact_matches": list(exact_matches),
            "similar_matches": [],
            "discount": 0.0,
        }

    # Rule 3: Partial match — check if missing skills have similar ones
    similar_matches = []
    unmatched = []

    for req_skill in missing_required:
        # Check if any candidate skill is similar to this required skill
        found_similar = False
        for cand_skill in candidate_set - exact_matches:
            if are_skills_similar(cand_skill, req_skill):
                similar_matches.append({
                    "required": req_skill,
                    "candidate_has": cand_skill,
                })
                found_similar = True
                break
        if not found_similar:
            unmatched.append(req_skill)

    # If all missing skills have similar matches → negotiate
    if len(unmatched) == 0:
        return {
            "eligible": True,
            "negotiable": True,
            "reason": f"Partial match with similar skills. Negotiate 10% discount.",
            "exact_matches": list(exact_matches),
            "similar_matches": similar_matches,
            "discount": 0.10,
        }

    # Some missing skills have no similar match → reject
    return {
        "eligible": False,
        "negotiable": False,
        "reason": f"Missing required skills with no similar match: {unmatched}.",
        "exact_matches": list(exact_matches),
        "similar_matches": [],
        "discount": 0.0,
    }
