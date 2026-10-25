import asyncio
import io
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import LETTER, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, KeepTogether, HRFlowable, Table, TableStyle
from google import genai
from google.genai import types
from groq import Groq
from pypdf import PdfReader

# Load environment variables from backend/.env
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = FastAPI(
    title="AI Resume Analyzer",
    description="Resume analysis, ATS matching, and resume optimization",
    version="2.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
# Keep the primary provider configurable; change this value in backend/.env if needed.
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
client = genai.Client(api_key=API_KEY) if API_KEY else None

# Backup provider: Groq. Create a key at https://console.groq.com/.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
groq_client = (
    Groq(api_key=GROQ_API_KEY, timeout=60.0, max_retries=1)
    if GROQ_API_KEY
    else None
)

MAX_PDF_SIZE = 5 * 1024 * 1024
MAX_RESUME_CHARS = 30000
MAX_JOB_CHARS = 15000
MAX_OPTIMIZED_RESUME_CHARS = 50000


def _validate_ai_response(response_text: str, json_mode: bool) -> str:
    """Reject empty output and malformed JSON so the backup can be tried."""
    if not response_text or not response_text.strip():
        raise ValueError("AI provider returned an empty response")

    cleaned = response_text.strip()
    if json_mode:
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("AI provider returned JSON that was not an object")

    return response_text.strip()


def call_gemini(prompt: str, json_mode: bool = False) -> str:
    """Call Gemini, retrying temporary server errors before returning failure."""
    if not API_KEY or client is None:
        raise RuntimeError("Gemini is not configured")

    started_at = time.perf_counter()
    config = None
    if json_mode:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=4096,
            temperature=0.2,
        )

    max_attempts = 4
    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config,
            )
            response_text = getattr(response, "text", None)
            response_text = _validate_ai_response(response_text or "", json_mode)
            print(
                f"Gemini request completed in {time.perf_counter() - started_at:.1f}s "
                f"(model={MODEL_NAME})."
            )
            return response_text
        except Exception as exc:
            lowered = str(exc).lower()
            retryable = any(term in lowered for term in (
                "503", "unavailable", "high demand", "overloaded",
                "500", "502", "504", "internal server error",
            ))
            if retryable and attempt < max_attempts - 1:
                wait_seconds = 2 ** (attempt + 1)
                print(
                    f"Gemini temporary error on attempt {attempt + 1}/{max_attempts}; "
                    f"retrying in {wait_seconds} seconds."
                )
                time.sleep(wait_seconds)
                continue
            raise

    raise RuntimeError("Gemini request failed after retries")


def call_groq(prompt: str, json_mode: bool = False) -> str:
    """Call Groq as the backup provider."""
    if not GROQ_API_KEY or groq_client is None:
        raise RuntimeError("Groq backup is not configured")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful resume analysis and resume optimization assistant. "
                "Never invent the candidate's skills, qualifications, employment, "
                "dates, achievements, or metrics. Follow the user's output format."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    request_options = {
        "model": GROQ_MODEL,
        "messages": messages,
        "max_completion_tokens": 4096,
    }
    if json_mode:
        # GPT-OSS 20B supports JSON Object Mode on Groq.
        request_options["response_format"] = {"type": "json_object"}

    started_at = time.perf_counter()
    response = groq_client.chat.completions.create(**request_options)
    response_text = response.choices[0].message.content or ""
    response_text = _validate_ai_response(response_text, json_mode)
    print(
        f"Groq fallback completed in {time.perf_counter() - started_at:.1f}s "
        f"(model={GROQ_MODEL})."
    )
    return response_text


def call_ai(prompt: str, json_mode: bool = False) -> tuple[str, str, str]:
    """Try Gemini first, then automatically fall back to Groq on any failure."""
    gemini_error = None

    if client is not None:
        try:
            return call_gemini(prompt, json_mode), "Gemini", MODEL_NAME
        except Exception as exc:
            gemini_error = exc
            print(
                f"Gemini failed ({type(exc).__name__}); "
                "trying the configured Groq backup if available."
            )
    else:
        gemini_error = RuntimeError("Gemini API key is not configured")
        print("Gemini is not configured; checking the Groq backup.")

    if groq_client is not None:
        try:
            return call_groq(prompt, json_mode), "Groq", GROQ_MODEL
        except Exception as groq_error:
            print(f"Groq fallback failed ({type(groq_error).__name__}).")
            raise HTTPException(
                status_code=502,
                detail=(
                    "Both AI providers failed to return a result. Please try again shortly. "
                    "If the problem continues, check the backend logs and both providers' "
                    "API keys, model access, and usage limits."
                ),
            ) from None

    if not client and not groq_client:
        raise HTTPException(
            status_code=503,
            detail=(
                "No AI provider is configured. Add GEMINI_API_KEY and/or GROQ_API_KEY "
                "to backend/.env, then restart the backend."
            ),
        ) from None

    # Gemini was configured but failed, and there is no backup key configured.
    print(f"Gemini failed and Groq is not configured ({type(gemini_error).__name__}).")
    raise HTTPException(
        status_code=502,
        detail=(
            "Gemini could not complete this request and the Groq backup is not configured. "
            "Add GROQ_API_KEY to backend/.env to enable automatic fallback, then restart."
        ),
    ) from None

def parse_json_response(text: str) -> dict:
    """Parse a JSON response, including one wrapped in Markdown fences."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        result = json.loads(cleaned)
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object.")
        return result
    except (json.JSONDecodeError, ValueError):
        print("AI provider returned an invalid JSON object.")
        raise HTTPException(
            status_code=502,
            detail=(
                "The AI response could not be formatted correctly. "
                "Please try again. Check the backend terminal if this repeats."
            ),
        ) from None


def validate_resume_inputs(resume_text: str, job_description: str):
    if not resume_text.strip() or not job_description.strip():
        raise HTTPException(
            status_code=400,
            detail="Resume text and job description are required.",
        )
    if len(resume_text) > MAX_RESUME_CHARS:
        raise HTTPException(status_code=413, detail="Resume text is too long.")
    if len(job_description) > MAX_JOB_CHARS:
        raise HTTPException(status_code=413, detail="Job description is too long.")


@app.get("/")
def home():
    return {
        "message": "AI Resume Analyzer backend is running!",
        "status": "success",
        "version": "2.2.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "ai_configured": bool(client or groq_client),
        "primary_provider": "Gemini" if client else None,
        "backup_provider_configured": bool(groq_client),
        "model": MODEL_NAME if client else (GROQ_MODEL if groq_client else None),
    }


@app.post("/extract-resume")
async def extract_resume(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF resume.")

    if file.content_type not in (None, "application/pdf"):
        raise HTTPException(status_code=400, detail="The uploaded file must be a PDF.")

    content = await file.read()
    if len(content) > MAX_PDF_SIZE:
        raise HTTPException(status_code=413, detail="PDF must be 5 MB or smaller.")

    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            raise HTTPException(
                status_code=400,
                detail="Password-protected PDFs are not supported.",
            )

        resume_text = "\n".join(
            page.extract_text() or "" for page in reader.pages
        ).strip()

        if not resume_text:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No readable text was found. Scanned PDFs may require OCR."
                ),
            )

        return {
            "filename": file.filename,
            "text": resume_text,
            "characters": len(resume_text),
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"PDF extraction error: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=400,
            detail="Could not read this PDF. Please try another file.",
        ) from None
    finally:
        await file.close()


@app.post("/analyze-resume")
async def analyze_resume(
    resume_text: str = Form(...),
    job_description: str = Form(...),
):
    validate_resume_inputs(resume_text, job_description)

    prompt = f"""
You are an experienced resume reviewer and career assistant.
Compare the candidate's resume against the supplied job description.
Use simple English and provide practical, honest feedback.

Return these sections:
1. Estimated overall match score from 0 to 100.
2. Resume strengths.
3. Skills matching the job.
4. Missing or weak skills.
5. Specific resume improvement suggestions.
6. Keywords to consider adding, only when truthful.
7. Five likely interview questions.

The score is an estimate, not a hiring prediction.
Never claim that a resume is guaranteed to pass an ATS.
Never invent qualifications, experience, certifications, metrics, or projects.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}
"""
    analysis, provider_used, model_used = call_ai(prompt)
    return {
        "status": "success",
        "analysis": analysis,
        "provider": provider_used,
        "model": model_used,
    }


@app.post("/optimize-resume")
async def optimize_resume(
    resume_text: str = Form(...),
    job_description: str = Form(...),
):
    """Analyze a resume and produce a job-tailored resume and recommendations."""
    validate_resume_inputs(resume_text, job_description)

    prompt = f"""
You are an accurate ATS resume editor. Tailor the resume to the job description.
Return one valid JSON object only, with exactly these keys and types:
{{
  "ats_match_score": 0,
  "score_explanation": "1-2 concise sentences",
  "resume_strengths": ["up to 4 short items"],
  "matching_skills": ["up to 10 skills"],
  "missing_skills": [{{"skill":"name","importance":"High, Medium, or Low","reason":"brief reason"}}],
  "keyword_suggestions": ["up to 8 truthful relevant keywords"],
  "improvement_suggestions": ["up to 5 actionable suggestions"],
  "recommended_projects": [{{"title":"project idea","description":"brief description","technologies":["technology"],"features":["feature"],"resume_bullet_example":"Clearly label as an example to use only after building the project"}}],
  "optimized_resume": "Complete rewritten resume as plain text",
  "changes_made": ["up to 6 concise changes"]
}}

Rules:
- Score from 0 to 100; it is an estimate, not a hiring prediction.
- Preserve the candidate's real contact details, education, employers, roles, dates, and projects.
- Never invent skills, qualifications, achievements, metrics, or work history.
- Use relevant job keywords only when supported by the original resume.
- Preserve the original resume structure and ALL meaningful details. Do not shorten the resume by removing sections, skills, experience bullets, project bullets, education details, dates, or metrics. You may improve wording and reorder content to match the job, but retain every factual item unless it is clearly duplicated. Output the complete resume, not a summary. Aim to preserve the original number of sections and roughly the same number of bullets. Keep each bullet specific and truthful.
- Missing skills are learning recommendations. Project ideas are future suggestions, not completed work.
- Keep all explanations brief. Do not return extra keys or Markdown fences.

ORIGINAL RESUME:
{resume_text}

TARGET JOB DESCRIPTION:
{job_description}
"""
    # Run the blocking SDK request in a worker thread so other API requests remain responsive.
    raw_response, provider_used, model_used = await asyncio.to_thread(call_ai, prompt, True)
    result = parse_json_response(raw_response)

    score = result.get("ats_match_score")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise HTTPException(
            status_code=502,
            detail="The AI returned an invalid ATS score. Please retry.",
        )
    result["ats_match_score"] = max(0, min(100, round(score)))

    if not isinstance(result.get("optimized_resume"), str):
        raise HTTPException(
            status_code=502,
            detail="The AI did not return an optimized resume. Please retry.",
        )
    if len(result["optimized_resume"]) > MAX_OPTIMIZED_RESUME_CHARS:
        raise HTTPException(
            status_code=502,
            detail="The AI returned a resume that is too long to export. Please retry.",
        )

    return {
        "status": "success",
        "provider": provider_used,
        "model": model_used,
        **result,
    }


@app.post("/download-optimized-resume")
async def download_optimized_resume(
    optimized_resume: str = Form(...),
):
    """Create a polished, ATS-friendly resume PDF with clear section hierarchy."""
    if not optimized_resume.strip():
        raise HTTPException(status_code=400, detail="The optimized resume is empty.")
    if len(optimized_resume) > MAX_OPTIMIZED_RESUME_CHARS:
        raise HTTPException(status_code=413, detail="The optimized resume is too long to export.")

    buffer = io.BytesIO()
    navy = colors.HexColor("#174A7C")
    dark = colors.HexColor("#202833")
    muted = colors.HexColor("#555D66")
    rule = colors.HexColor("#2D5F91")

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.52 * inch,
        leftMargin=0.52 * inch,
        topMargin=0.38 * inch,
        bottomMargin=0.42 * inch,
        title="Optimized Resume",
        author="AI Resume Analyzer",
        pageCompression=1,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="CandidateName", parent=styles["Title"],
        fontName="Helvetica-Bold", fontSize=16, leading=18,
        textColor=navy, alignment=TA_CENTER, spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="ContactLine", parent=styles["Normal"],
        fontName="Helvetica", fontSize=7.5, leading=9,
        textColor=muted, alignment=TA_CENTER, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="SectionTitle", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=9.3, leading=11,
        textColor=navy, spaceBefore=5, spaceAfter=2,
        keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="ResumeBody2", parent=styles["BodyText"],
        fontName="Helvetica", fontSize=8.2, leading=10.3,
        textColor=dark, spaceAfter=2.1, alignment=TA_LEFT,
        splitLongWords=True, allowWidows=0, allowOrphans=0,
    ))
    styles.add(ParagraphStyle(
        name="ResumeBullet2", parent=styles["ResumeBody2"],
        leftIndent=10, firstLineIndent=-7, bulletIndent=0,
        spaceAfter=1.8,
    ))
    styles.add(ParagraphStyle(
        name="EntryTitle", parent=styles["ResumeBody2"],
        fontName="Helvetica-Bold", fontSize=8.4, leading=10.2,
        spaceBefore=2.5, spaceAfter=1, keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="TechLine", parent=styles["ResumeBody2"],
        fontName="Helvetica-Oblique", fontSize=7.8, leading=9.2,
        textColor=muted, spaceAfter=1.5, keepWithNext=True,
    ))

    section_names = {
        "professional summary", "summary", "profile", "career objective",
        "technical skills", "skills", "core skills", "professional experience",
        "work experience", "experience", "employment history", "internships",
        "projects", "personal projects", "education", "certifications",
        "achievements", "awards", "publications", "languages",
    }

    def clean_markdown(line: str) -> str:
        line = line.strip()
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^\s*[-*+•●▪?]\s*", "- ", line)
        # Convert simple inline Markdown emphasis to PDF paragraph markup.
        line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        line = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", line)
        line = line.replace("—", "-").replace("–", "-").replace("’", "'").replace("“", '"').replace("”", '"').replace("\u00a0", " ")
        # Use ASCII-safe punctuation because built-in Helvetica does not cover all Unicode glyphs.
        line = line.replace("•", "-").replace("●", "-").replace("▪", "-").replace("? ", "- ")
        line = line.encode("latin-1", "replace").decode("latin-1")
        return line

    def plain(line: str) -> str:
        return re.sub(r"</?(?:b|i)>", "", line).strip().rstrip(":").lower()

    raw_lines = [line.rstrip() for line in optimized_resume.replace("\r", "").split("\n")]
    lines = [line.strip() for line in raw_lines if line.strip()]
    if not lines:
        raise HTTPException(status_code=400, detail="The optimized resume is empty.")

    story = []
    first = clean_markdown(lines[0])
    story.append(Paragraph(first, styles["CandidateName"]))

    # The next one or two short lines are usually contact details.
    index = 1
    contact_parts = []
    while index < len(lines) and index < 4:
        current = lines[index].strip()
        if len(current) <= 180 and (
            "@" in current or re.search(r"\+?\d[\d\s().-]{7,}", current)
            or "linkedin" in current.lower() or "github" in current.lower()
            or "portfolio" in current.lower() or "bengaluru" in current.lower()
            or "india" in current.lower() or "|" in current
        ):
            contact_parts.append(clean_markdown(current))
            index += 1
        else:
            break
    if contact_parts:
        story.append(Paragraph(" &nbsp; | &nbsp; ".join(contact_parts), styles["ContactLine"]))
    story.append(HRFlowable(width="100%", thickness=1.1, color=rule, spaceBefore=2, spaceAfter=3))

    while index < len(lines):
        original = lines[index].strip()
        normalized = plain(original)
        cleaned = clean_markdown(original)

        # Section heading: render in blue with a fine rule, like a professional resume.
        heading_key = normalized.rstrip(":")
        if heading_key in section_names or (
            original.endswith(":") and len(original) < 48
        ):
            story.append(Paragraph(clean_markdown(heading_key.upper()), styles["SectionTitle"]))
            story.append(HRFlowable(width="100%", thickness=0.45, color=rule, spaceBefore=0, spaceAfter=3))
            index += 1
            continue

        # Bullet points are indented and spaced consistently.
        if cleaned.startswith("- "):
            story.append(Paragraph("-&nbsp;&nbsp;" + cleaned[2:], styles["ResumeBullet2"]))
            index += 1
            continue

        # Short standalone lines following an entry title are often tech stacks or location.
        next_line = clean_markdown(lines[index + 1]) if index + 1 < len(lines) else ""
        lower = original.lower()
        if (
            len(original) < 150
            and (original.count(",") >= 2 or lower.startswith(("technologies:", "tech stack:", "tools:")))
            and not original.endswith(".")
        ):
            story.append(Paragraph(cleaned, styles["TechLine"]))
            index += 1
            continue

        # Put common date strings on the right of an entry title when recognizable.
        date_match = re.search(
            r"^(.*?)(?:\s{2,}|\s+[|]\s+)(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s*[-–]\s*(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+)?\d{4}|\b\d{4}\s*[-–]\s*\d{4}|\b(?:19|20)\d{2}\b)$",
            original, flags=re.IGNORECASE,
        )
        if date_match and len(date_match.group(1).strip()) > 2:
            left = clean_markdown(date_match.group(1).strip())
            right = clean_markdown(date_match.group(2).strip())
            table = Table(
                [[Paragraph(left, styles["EntryTitle"]), Paragraph(right, ParagraphStyle(
                    name="DateRight", parent=styles["EntryTitle"], alignment=TA_RIGHT
                ))]],
                colWidths=[doc.width * 0.77, doc.width * 0.23],
                hAlign="LEFT",
            )
            table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]))
            story.append(table)
        elif len(original) < 110 and (
            index == 0 or
            (next_line and (
                next_line.startswith("•") or "," in lines[index + 1]
                or re.search(r"\b(19|20)\d{2}\b", lines[index + 1])
            ))
        ):
            story.append(Paragraph(cleaned, styles["EntryTitle"]))
        else:
            story.append(Paragraph(cleaned, styles["ResumeBody2"]))
        index += 1

    def add_page_number(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(muted)
        canvas.drawRightString(A4[0] - 0.52 * inch, 0.22 * inch, f"{document.page}")
        canvas.restoreState()

    try:
        doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    except Exception as exc:
        print(f"PDF generation error: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Could not generate the PDF. Please try downloading again.",
        ) from None

    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'attachment; filename="optimized_resume.pdf"',
            "Cache-Control": "no-store",
        },
    )
