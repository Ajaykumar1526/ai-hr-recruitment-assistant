"""
AI Recruitment Agent
----------------------
An agent built on Claude's native tool-use (function calling). Given a
free-form HR question about an already-loaded candidate + job description
(e.g. "Does this candidate fit? What should I ask them about Docker?"),
the agent decides which underlying tool(s) to call: skill matching,
knowledge-base retrieval, or interview question generation — then answers.

This is the "reasoning" layer on top of the deterministic pipeline in
app.py. The Streamlit UI's main analysis flow calls the pipeline functions
directly for speed/reliability; this agent is exposed as an optional
"Ask the Recruitment Agent" chat panel for follow-up questions.
"""

import json
from typing import Optional

from anthropic import Anthropic

from src.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, require_api_key
from src.matcher import compute_match
from src.rag import retrieve_knowledge
from src.interview_generator import generate_interview_questions


AGENT_SYSTEM_PROMPT = """
You are an AI Recruitment Agent helping an HR professional evaluate a specific
candidate against a specific job description that is already loaded into the
system. Use the available tools whenever they would give you real data instead
of guessing:
- recompute_match: get the latest skill-match breakdown and score
- retrieve_recruitment_knowledge: look up interview guidelines, skill definitions,
  or recruitment best practices relevant to the question
- generate_interview_questions: produce a fresh, targeted interview question set

Never base any judgment on gender, race, religion, age, disability, or other
protected characteristics. Keep answers concise, specific, and grounded in the
tool results and the candidate/job data you were given. If a question is outside
the scope of this candidate/job, say so briefly.
"""

TOOLS = [
    {
        "name": "recompute_match",
        "description": "Recompute the candidate-to-job match score and skill breakdown "
                        "(strong/partial/missing) using the loaded resume and job description.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "retrieve_recruitment_knowledge",
        "description": "Search the recruitment knowledge base (interview guidelines, skill "
                        "definitions, best practices, role requirement baselines) for guidance "
                        "relevant to a query.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to search for"}},
            "required": ["query"],
        },
    },
    {
        "name": "generate_interview_questions",
        "description": "Generate a fresh set of technical, behavioral, and role-specific "
                        "interview questions for the loaded candidate and job.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


def _execute_tool(tool_name: str, tool_input: dict, resume: dict, jd: dict) -> str:
    """Run the actual Python function behind a tool the agent decided to call."""
    if tool_name == "recompute_match":
        result = compute_match(resume, jd)
        return json.dumps(result)

    if tool_name == "retrieve_recruitment_knowledge":
        query = tool_input.get("query", "")
        chunks = retrieve_knowledge(query, n_results=3)
        return json.dumps(chunks)

    if tool_name == "generate_interview_questions":
        match_result = compute_match(resume, jd)
        questions = generate_interview_questions(resume, jd, match_result)
        return json.dumps(questions)

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def ask_agent(user_question: str, resume: dict, jd: dict,
               client: Optional[Anthropic] = None, max_turns: int = 4) -> str:
    """
    Run an agentic loop: Claude decides which tool(s) to call (if any) to
    answer the HR user's question about the loaded candidate/job, then
    produces a final natural-language answer.
    """
    require_api_key()
    client = client or Anthropic(api_key=ANTHROPIC_API_KEY)

    context_summary = (
        f"Loaded candidate: {resume.get('name', 'Unknown')} "
        f"(skills: {', '.join(resume.get('skills', []) or [])})\n"
        f"Loaded job: {jd.get('job_title', 'Unknown')} "
        f"(required skills: {', '.join(jd.get('required_skills', []) or [])})"
    )

    messages = [
        {"role": "user", "content": f"{context_summary}\n\nHR question: {user_question}"}
    ]

    for _ in range(max_turns):
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1500,
            system=AGENT_SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            return "".join(block.text for block in response.content if block.type == "text").strip()

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result_text = _execute_tool(block.name, block.input, resume, jd)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })
        messages.append({"role": "user", "content": tool_results})

    return "I wasn't able to finish reasoning about that within the allotted steps. Please try a more specific question."
