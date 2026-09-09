"""
Interview Question Generator
------------------------------
Generates personalized technical, behavioral, and role-specific interview
questions based on the candidate's skills/projects, the job requirements,
and their skill gaps — grounded with retrieved best-practice guidance (RAG).
"""

import json
from typing import Optional

from anthropic import Anthropic

from src.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, require_api_key
from src.resume_parser import _safe_json_parse
from src.rag import retrieve_knowledge


QUESTION_GEN_SYSTEM_PROMPT = """
You are an expert technical interviewer creating a personalized interview plan.
Base your questions only on the candidate/job information and interview
guidance context provided. Do not reference or infer protected characteristics
(age, gender, religion, disability, etc.).

Return STRICT JSON only, no markdown fences, no commentary, with this schema:
{
  "technical_questions": [
    {"question": string, "based_on": string, "evaluation_points": [string]}
  ],
  "behavioral_questions": [
    {"question": string, "based_on": string, "evaluation_points": [string]}
  ],
  "role_specific_questions": [
    {"question": string, "based_on": string, "evaluation_points": [string]}
  ]
}

Rules:
- Generate 3 technical, 2 behavioral, and 2 role-specific questions.
- "based_on" should briefly say what prompted the question (e.g. a specific
  project, a missing skill, or a job responsibility).
- Include at least one technical question probing a missing/partial-match skill,
  framed as gauging ramp-up ability rather than penalizing the gap.
- "evaluation_points" should be 2-3 concrete things a strong answer would include.
- Return ONLY the JSON object.
"""


def generate_interview_questions(resume: dict, jd: dict, match_result: dict,
                                  client: Optional[Anthropic] = None) -> dict:
    """Generate a personalized interview question set, grounded with RAG context."""
    require_api_key()
    client = client or Anthropic(api_key=ANTHROPIC_API_KEY)

    # Pull relevant guidance from the knowledge base (RAG retrieval)
    guidance_chunks = retrieve_knowledge(
        f"interview questions for {jd.get('job_title', 'this role')} covering "
        f"{', '.join((jd.get('required_skills') or [])[:5])}",
        n_results=3,
    )
    guidance_text = "\n\n".join(f"[{c['source']}] {c['text']}" for c in guidance_chunks)

    projects = resume.get("projects", []) or []
    project_summaries = "; ".join(
        f"{p.get('name', '')}: {p.get('description', '')}" for p in projects
    )

    context = f"""
Candidate name: {resume.get('name', 'Unknown')}
Candidate skills: {', '.join(resume.get('skills', []) or [])}
Candidate projects: {project_summaries}
Candidate experience: {match_result.get('candidate_years', 0)} years

Job title: {jd.get('job_title', 'Unknown')}
Job responsibilities: {'; '.join(jd.get('responsibilities', []) or [])}
Required skills: {', '.join(jd.get('required_skills', []) or [])}

Strong-match skills: {', '.join(match_result['required_skills']['strong_match'])}
Partial-match skills: {', '.join(match_result['required_skills']['partial_match'])}
Missing required skills: {', '.join(match_result['required_skills']['missing'])}

Relevant interview guidance (from knowledge base):
{guidance_text}
"""

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=2000,
        system=QUESTION_GEN_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}],
    )

    raw_text = "".join(block.text for block in response.content if block.type == "text")
    return _safe_json_parse(raw_text)
