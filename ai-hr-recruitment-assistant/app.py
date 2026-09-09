"""
AI HR Recruitment Assistant — Streamlit App
---------------------------------------------
Upload a resume, provide a job description, and get an AI-powered match
score, skill-gap analysis, personalized interview questions, a recruitment
recommendation, and a downloadable PDF report.

Run with:  streamlit run app.py
"""

import streamlit as st

from src.config import ANTHROPIC_API_KEY
from src.resume_parser import parse_resume
from src.jd_parser import parse_job_description, extract_jd_text_from_file
from src.matcher import compute_match, generate_match_explanation, generate_recommendation
from src.interview_generator import generate_interview_questions
from src.rag import build_knowledge_base
from src.report_generator import build_report_pdf
from src.agent import ask_agent


st.set_page_config(page_title="AI HR Recruitment Assistant", page_icon="🧑‍💼", layout="wide")


# ---------- Session state ----------
for key in ["resume", "jd", "match_result", "explanation", "questions", "recommendation", "chat_history"]:
    if key not in st.session_state:
        st.session_state[key] = None
if st.session_state["chat_history"] is None:
    st.session_state["chat_history"] = []


def reset_analysis():
    for key in ["match_result", "explanation", "questions", "recommendation"]:
        st.session_state[key] = None


# ---------- Sidebar ----------
with st.sidebar:
    st.title("🧑‍💼 AI HR Recruitment Assistant")
    st.caption("Resume screening, JD matching, skill-gap analysis, and interview prep — powered by Claude.")

    if not ANTHROPIC_API_KEY:
        st.error("ANTHROPIC_API_KEY not set. Copy `.env.example` to `.env` and add your key.")

    st.divider()
    st.subheader("1. Upload Resume")
    resume_file = st.file_uploader("PDF or DOCX resume", type=["pdf", "docx"], key="resume_uploader")

    st.subheader("2. Job Description")
    jd_input_mode = st.radio("Provide JD as:", ["Paste text", "Upload file"], horizontal=True)
    jd_text_input, jd_file = "", None
    if jd_input_mode == "Paste text":
        jd_text_input = st.text_area("Paste the job description", height=200)
    else:
        jd_file = st.file_uploader("PDF or DOCX job description", type=["pdf", "docx"], key="jd_uploader")

    analyze_clicked = st.button("🔍 Analyze Candidate", type="primary", use_container_width=True)


# ---------- Analysis pipeline ----------
if analyze_clicked:
    if not ANTHROPIC_API_KEY:
        st.error("Please configure your ANTHROPIC_API_KEY first.")
    elif not resume_file:
        st.warning("Please upload a resume.")
    elif jd_input_mode == "Paste text" and not jd_text_input.strip():
        st.warning("Please paste a job description.")
    elif jd_input_mode == "Upload file" and not jd_file:
        st.warning("Please upload a job description file.")
    else:
        reset_analysis()
        with st.spinner("Parsing resume..."):
            resume_data = parse_resume(resume_file.read(), resume_file.name)
            st.session_state["resume"] = resume_data

        with st.spinner("Parsing job description..."):
            if jd_input_mode == "Upload file":
                jd_raw_text = extract_jd_text_from_file(jd_file.read(), jd_file.name)
            else:
                jd_raw_text = jd_text_input
            jd_data = parse_job_description(jd_raw_text)
            st.session_state["jd"] = jd_data

        with st.spinner("Preparing recruitment knowledge base..."):
            build_knowledge_base()

        with st.spinner("Computing match score and skill gaps..."):
            match_result = compute_match(resume_data, jd_data)
            st.session_state["match_result"] = match_result

        with st.spinner("Generating match explanation..."):
            st.session_state["explanation"] = generate_match_explanation(resume_data, jd_data, match_result)

        with st.spinner("Generating interview questions..."):
            st.session_state["questions"] = generate_interview_questions(resume_data, jd_data, match_result)

        with st.spinner("Generating recommendation..."):
            st.session_state["recommendation"] = generate_recommendation(resume_data, jd_data, match_result)

        st.success("Analysis complete!")


# ---------- Results ----------
resume = st.session_state["resume"]
jd = st.session_state["jd"]
match_result = st.session_state["match_result"]

if not resume or not jd or not match_result:
    st.info("Upload a resume and a job description in the sidebar, then click **Analyze Candidate**.")
    st.stop()

tab_overview, tab_skills, tab_questions, tab_recommendation, tab_agent, tab_report = st.tabs(
    ["📋 Overview", "🧩 Skill Gap", "❓ Interview Questions", "✅ Recommendation", "🤖 Ask the Agent", "📄 Report"]
)

with tab_overview:
    col1, col2, col3 = st.columns(3)
    col1.metric("Match Score", f"{match_result['match_score']} / 100")
    col2.metric("Semantic Fit", f"{match_result['semantic_similarity']}%")
    col3.metric("Experience Fit", f"{match_result['experience_fit_pct']}%")

    st.subheader(f"Candidate: {resume.get('name', 'Unknown')}")
    left, right = st.columns(2)
    with left:
        st.markdown("**Resume Summary**")
        st.write(resume.get("summary", "—"))
        st.markdown(f"**Total Experience:** {resume.get('total_experience_years', 0)} years")
        st.markdown(f"**Skills:** {', '.join(resume.get('skills', []) or [])}")
        with st.expander("Education"):
            for edu in resume.get("education", []) or []:
                st.write(f"- {edu.get('degree')} — {edu.get('institution')} ({edu.get('year')}) {edu.get('score')}")
        with st.expander("Experience"):
            for exp in resume.get("experience", []) or []:
                st.write(f"- **{exp.get('title')}** at {exp.get('company')} ({exp.get('duration')}): {exp.get('description')}")
        with st.expander("Projects"):
            for proj in resume.get("projects", []) or []:
                st.write(f"- **{proj.get('name')}**: {proj.get('description')} — _{', '.join(proj.get('technologies', []))}_")
        with st.expander("Certifications"):
            for cert in resume.get("certifications", []) or []:
                st.write(f"- {cert}")

    with right:
        st.markdown("**Job Description**")
        st.markdown(f"**Title:** {jd.get('job_title', '—')}  |  **Company:** {jd.get('company', '—')}")
        st.markdown(f"**Min. Experience:** {jd.get('min_experience_years', 0)} years")
        st.markdown(f"**Required Skills:** {', '.join(jd.get('required_skills', []) or [])}")
        st.markdown(f"**Preferred Skills:** {', '.join(jd.get('preferred_skills', []) or [])}")
        with st.expander("Responsibilities"):
            for r in jd.get("responsibilities", []) or []:
                st.write(f"- {r}")
        with st.expander("Qualifications"):
            for q in jd.get("qualifications", []) or []:
                st.write(f"- {q}")

    st.subheader("Match Explanation")
    st.write(st.session_state["explanation"])

with tab_skills:
    req = match_result["required_skills"]
    pref = match_result["preferred_skills"]

    st.subheader("Required Skills")
    c1, c2, c3 = st.columns(3)
    c1.success("**Strong Match**\n\n" + ("\n".join(f"- {s}" for s in req["strong_match"]) or "None"))
    c2.warning("**Partial Match**\n\n" + ("\n".join(f"- {s}" for s in req["partial_match"]) or "None"))
    c3.error("**Missing**\n\n" + ("\n".join(f"- {s}" for s in req["missing"]) or "None"))

    st.subheader("Preferred Skills")
    c4, c5, c6 = st.columns(3)
    c4.success("**Strong Match**\n\n" + ("\n".join(f"- {s}" for s in pref["strong_match"]) or "None"))
    c5.warning("**Partial Match**\n\n" + ("\n".join(f"- {s}" for s in pref["partial_match"]) or "None"))
    c6.error("**Missing**\n\n" + ("\n".join(f"- {s}" for s in pref["missing"]) or "None"))

with tab_questions:
    questions = st.session_state["questions"] or {}
    for section_title, key in [
        ("Technical Questions", "technical_questions"),
        ("Behavioral Questions", "behavioral_questions"),
        ("Role-Specific Questions", "role_specific_questions"),
    ]:
        st.subheader(section_title)
        for q in questions.get(key, []) or []:
            with st.container(border=True):
                st.markdown(f"**Q:** {q.get('question')}")
                st.caption(f"Based on: {q.get('based_on')}")
                st.markdown("Evaluation points:")
                for point in q.get("evaluation_points", []) or []:
                    st.markdown(f"- {point}")

with tab_recommendation:
    rec = st.session_state["recommendation"] or {}
    decision = rec.get("decision", "N/A")
    color_map = {
        "Strongly Recommend": "success", "Recommend": "success",
        "Consider": "warning", "Not Recommended": "error",
    }
    getattr(st, color_map.get(decision, "info"))(f"### {decision}")
    st.write(rec.get("explanation", ""))

with tab_agent:
    st.subheader("Ask the AI Recruitment Agent")
    st.caption("Ask follow-up questions — the agent decides whether to recompute the match, "
               "search recruitment knowledge, or generate new interview questions.")

    for role, msg in st.session_state["chat_history"]:
        with st.chat_message(role):
            st.write(msg)

    user_q = st.chat_input("e.g. 'Focus interview questions on their SQL gap'")
    if user_q:
        st.session_state["chat_history"].append(("user", user_q))
        with st.spinner("Agent is reasoning..."):
            answer = ask_agent(user_q, resume, jd)
        st.session_state["chat_history"].append(("assistant", answer))
        st.rerun()

with tab_report:
    st.subheader("Downloadable Recruitment Report")
    if st.button("📄 Generate PDF Report"):
        with st.spinner("Building report..."):
            pdf_bytes = build_report_pdf(
                resume, jd, match_result,
                st.session_state["explanation"],
                st.session_state["questions"],
                st.session_state["recommendation"],
            )
        st.download_button(
            "⬇️ Download Report",
            data=pdf_bytes,
            file_name=f"{resume.get('name', 'candidate').replace(' ', '_')}_recruitment_report.pdf",
            mime="application/pdf",
        )
