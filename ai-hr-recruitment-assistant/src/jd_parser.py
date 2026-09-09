"""
Job Description Parser
-----------------------
Takes a raw job description (typed or uploaded) and extracts structured
requirements: required skills, preferred skills, qualifications, experience
level, and responsibilities.
"""

from typing import Optional

from anthropic import Anthropic

from src.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, require_api_key
from src.resume_parser import extract_text_from_pdf, extract_text_from_docx, _safe_json_parse


JD_SCHEMA_INSTRUCTIONS = """
You are an expert job description analyst for an HR recruitment system.

Read the job description text below and extract it into STRICT JSON only
(no markdown fences, no commentary, no extra text).

Use exactly this schema:
{
  "job_title": string,
  "company": string,
  "required_skills": [string],
  "preferred_skills": [string],
  "qualifications": [string],
  "min_experience_years": number,
  "responsibilities": [string],
  "employment_type": string,
  "location": string
}

Rules:
- "required_skills" are must-have/mandatory skills explicitly stated as required.
- "preferred_skills" are nice-to-have or bonus skills.
- If a field is missing, use an empty string, empty list, or 0.
- Return ONLY the JSON object, nothing else.
"""


def extract_jd_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Extract text from an uploaded JD file (PDF or DOCX)."""
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif lower_name.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or DOCX, or paste text instead.")


def parse_job_description(jd_text: str, client: Optional[Anthropic] = None) -> dict:
    """Send JD text to Claude and get back structured JSON requirements."""
    require_api_key()
    if not jd_text.strip():
        raise ValueError("Job description text is empty.")

    client = client or Anthropic(api_key=ANTHROPIC_API_KEY)

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1500,
        system=JD_SCHEMA_INSTRUCTIONS,
        messages=[{"role": "user", "content": f"Job description text:\n\n{jd_text[:12000]}"}],
    )

    raw_text = "".join(block.text for block in response.content if block.type == "text")
    structured = _safe_json_parse(raw_text)
    structured["_raw_text"] = jd_text
    return structured
