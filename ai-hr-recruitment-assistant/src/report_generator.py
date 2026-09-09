"""
Recruitment Report Generator
-------------------------------
Builds a downloadable PDF report summarizing the candidate, job, match
score, skill-gap breakdown, interview questions, and recommendation.
"""

import io
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem
)


def _skill_list(skills):
    return ", ".join(skills) if skills else "None"


def build_report_pdf(resume: dict, jd: dict, match_result: dict,
                      explanation: str, questions: dict, recommendation: dict) -> bytes:
    """Render all analysis results into a polished PDF and return its bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm, leftMargin=2 * cm, rightMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=20, spaceAfter=6)
    h2 = ParagraphStyle("H2Custom", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6,
                         textColor=colors.HexColor("#1f2937"))
    body = styles["BodyText"]

    elements = []
    elements.append(Paragraph("AI HR Recruitment Assistant — Candidate Report", title_style))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body))
    elements.append(Spacer(1, 12))

    # Candidate & Job summary
    elements.append(Paragraph("Candidate & Role Overview", h2))
    summary_table_data = [
        ["Candidate", resume.get("name", "Unknown")],
        ["Candidate Experience", f"{match_result.get('candidate_years', 0)} years"],
        ["Job Title", jd.get("job_title", "Unknown")],
        ["Required Experience", f"{match_result.get('min_required_years', 0)} years"],
        ["Match Score", f"{match_result.get('match_score', 0)} / 100"],
    ]
    table = Table(summary_table_data, colWidths=[5 * cm, 10 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f3f4f6")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(table)

    # Match explanation
    elements.append(Paragraph("Match Explanation", h2))
    elements.append(Paragraph(explanation or "No explanation generated.", body))

    # Skill gap analysis
    elements.append(Paragraph("Skill Gap Analysis (Required Skills)", h2))
    req = match_result.get("required_skills", {})
    elements.append(Paragraph(f"<b>Strong Match:</b> {_skill_list(req.get('strong_match'))}", body))
    elements.append(Paragraph(f"<b>Partial Match:</b> {_skill_list(req.get('partial_match'))}", body))
    elements.append(Paragraph(f"<b>Missing:</b> {_skill_list(req.get('missing'))}", body))

    # Interview questions
    elements.append(Paragraph("Suggested Interview Questions", h2))
    for section_title, key in [
        ("Technical", "technical_questions"),
        ("Behavioral", "behavioral_questions"),
        ("Role-Specific", "role_specific_questions"),
    ]:
        q_list = questions.get(key, []) if questions else []
        if not q_list:
            continue
        elements.append(Paragraph(f"<b>{section_title}</b>", body))
        items = []
        for q in q_list:
            eval_points = "; ".join(q.get("evaluation_points", []))
            items.append(ListItem(Paragraph(
                f"{q.get('question', '')}<br/><i>Look for: {eval_points}</i>", body
            )))
        elements.append(ListFlowable(items, bulletType="bullet"))
        elements.append(Spacer(1, 6))

    # Recommendation
    elements.append(Paragraph("Recruitment Recommendation", h2))
    elements.append(Paragraph(f"<b>{recommendation.get('decision', 'N/A')}</b>", body))
    elements.append(Paragraph(recommendation.get("explanation", ""), body))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
