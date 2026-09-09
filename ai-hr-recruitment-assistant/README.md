# AI HR Recruitment Assistant

An AI-powered recruitment assistant that helps HR professionals screen resumes,
analyze job descriptions, match candidates to roles, identify skill gaps,
generate personalized interview questions, and produce a downloadable
recruitment report — all from a single Streamlit dashboard.

Built with: **Streamlit + Claude (Anthropic API) + RAG (ChromaDB) + Sentence
Transformers + an AI agent with tool-use.**

---

## Features

- **Resume Upload & Parsing** — Upload a PDF/DOCX resume; Claude extracts name,
  education, experience, skills, certifications, and projects into structured data.
- **Job Description Analysis** — Paste or upload a JD; Claude extracts required
  skills, preferred skills, qualifications, experience, and responsibilities.
- **Candidate-JD Matching** — A 0–100 match score blending semantic similarity,
  skill overlap, and experience fit, with a plain-language explanation.
- **Skill Gap Analysis** — Skills categorized as Strong Match / Partial Match /
  Missing, using sentence-embedding similarity (not just exact string matching).
- **AI Recruitment Agent** — A Claude tool-use agent you can chat with for
  follow-up questions; it decides whether to recompute the match, search the
  knowledge base, or regenerate interview questions.
- **Interview Question Generator** — Personalized technical, behavioral, and
  role-specific questions, grounded in a RAG knowledge base of interview
  guidelines, with evaluation points for each question.
- **Recruitment Recommendation** — Strongly Recommend / Recommend / Consider /
  Not Recommended, with an explanation based strictly on skills and experience
  (never on protected characteristics).
- **RAG Knowledge Base** — Markdown docs (interview guidelines, skill
  definitions, best practices, role baselines) embedded and retrieved via
  ChromaDB + Sentence Transformers.
- **Downloadable Report** — A polished PDF report summarizing everything above.

---

## Project Structure

```
ai-hr-recruitment-assistant/
│
├── app.py                     # Streamlit UI (entry point)
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── src/
│   ├── config.py               # Loads settings from .env
│   ├── agent.py                 # AI agent (Claude tool-use)
│   ├── resume_parser.py         # PDF/DOCX -> structured resume JSON
│   ├── jd_parser.py             # JD text/file -> structured JD JSON
│   ├── matcher.py                # Match score, explanation, recommendation
│   ├── skill_extractor.py        # Embedding-based skill categorization
│   ├── interview_generator.py    # Personalized interview questions
│   ├── rag.py                     # Knowledge base ingestion + retrieval
│   └── report_generator.py       # PDF report builder
│
├── data/
│   ├── knowledge_base/           # RAG source documents (.md)
│   │   ├── interview_guidelines.md
│   │   ├── skill_definitions.md
│   │   ├── recruitment_best_practices.md
│   │   └── role_requirements.md
│   └── sample_data/              # Sample resume + JD for testing
│       ├── sample_resume.txt
│       └── sample_job_description.txt
│
├── uploads/                     # Uploaded files land here at runtime
└── reports/                     # Generated PDF reports (optional local copies)
```

---

## How It Works

1. **Parsing** (`resume_parser.py`, `jd_parser.py`) — `PyMuPDF` and
   `python-docx` extract raw text from uploaded files. Claude then converts
   that raw text into strict JSON following a fixed schema.
2. **Matching** (`matcher.py`, `skill_extractor.py`) — `sentence-transformers`
   embeds resume text, JD text, and individual skills. Cosine similarity gives
   both an overall semantic-fit score and per-skill match categories
   (Strong / Partial / Missing). These are blended with an experience-fit
   score into a final 0–100 match score.
3. **RAG Knowledge Base** (`rag.py`) — Markdown files in
   `data/knowledge_base/` are chunked and embedded into a persistent
   **ChromaDB** collection on first run. `retrieve_knowledge()` fetches the
   most relevant chunks for a query (e.g. when generating interview
   questions), so outputs are grounded in real recruitment guidance instead
   of the model's unaided guesses.
4. **AI Agent** (`agent.py`) — Uses Claude's native tool-use (function
   calling). Given a free-form HR question, the agent decides which tool to
   call — `recompute_match`, `retrieve_recruitment_knowledge`, or
   `generate_interview_questions` — executes it, and reasons over the result
   before answering. This is exposed as the **"Ask the Agent"** tab.
5. **Interview Questions** (`interview_generator.py`) — Combines the
   candidate's skills/projects, the JD's requirements, the skill-gap
   breakdown, and retrieved interview guidance to generate technical,
   behavioral, and role-specific questions with evaluation points.
6. **Report** (`report_generator.py`) — `reportlab` renders all of the above
   into a downloadable PDF from the **Report** tab.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ajaykumar1526/ai-hr-recruitment-assistant.git
cd ai-hr-recruitment-assistant
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

```bash
cp .env.example .env
```

Open `.env` and set your Anthropic API key (get one at
https://console.anthropic.com/):

```
ANTHROPIC_API_KEY=your_actual_key_here
```

> **Never commit your real `.env` file.** It's already excluded via
> `.gitignore`.

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. The first run will download the
sentence-transformer embedding model (~90MB) and build the local RAG
knowledge base — this only happens once.

---

## Usage

1. Upload a candidate's resume (PDF or DOCX) in the sidebar.
2. Paste or upload a job description.
3. Click **Analyze Candidate**.
4. Explore the tabs:
   - **Overview** — parsed resume + JD side by side, match score, explanation
   - **Skill Gap** — Strong/Partial/Missing skills for required & preferred
   - **Interview Questions** — personalized technical/behavioral/role questions
   - **Recommendation** — final hiring recommendation with reasoning
   - **Ask the Agent** — chat with the AI agent for follow-up questions
   - **Report** — generate and download a PDF summary

To try it quickly without your own files, use the sample resume and job
description in `data/sample_data/` (save them as `.docx`/`.pdf` first, or
paste the JD text directly into the sidebar).

---

## Fairness Note

Match scores, skill-gap analysis, and recommendations are generated using
only skills, experience, qualifications, and job requirements. Prompts
explicitly instruct the model not to consider gender, race, religion, age,
disability, or other protected characteristics. This system is a **decision
support tool** — final hiring decisions should always involve human review.

---

## Troubleshooting

- **"ANTHROPIC_API_KEY is not set"** — Make sure you copied `.env.example` to
  `.env` and filled in a real key.
- **Slow first run** — The embedding model and knowledge base are built once
  and cached; subsequent runs are much faster.
- **"Could not parse structured data from the model response"** — Rare model
  formatting hiccup; try re-running the analysis.
- **Chroma/embedding errors on first install** — Ensure you're on Python 3.9+
  and that `pip install -r requirements.txt` completed without errors.

---

## Tech Stack

Python · Streamlit · Anthropic API (Claude) · ChromaDB · Sentence
Transformers · PyMuPDF · python-docx · ReportLab
