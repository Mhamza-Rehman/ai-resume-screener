"""Shared helpers for the resume screening application."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import pdfplumber
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


SUPPORTED_RESUME_MIME_TYPES: tuple[str, ...] = ("application/pdf",)
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"


class EducationEntry(BaseModel):
    institution: str = ""
    degree: str = ""
    field_of_study: str = ""
    start_year: str = ""
    end_year: str = ""
    notes: str = ""


class ExperienceEntry(BaseModel):
    company: str = ""
    role: str = ""
    start_date: str = ""
    end_date: str = ""
    years: float | None = None
    highlights: list[str] = Field(default_factory=list)


class ProjectEntry(BaseModel):
    name: str = ""
    description: str = ""
    technologies: list[str] = Field(default_factory=list)


class CandidateProfile(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    skills: list[str] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)


@dataclass(frozen=True)
class ValidationResult:
    """Represents a safe validation outcome for user inputs."""

    is_valid: bool
    message: str = ""


def initialize_session_state(session_state: dict) -> None:
    """Seed Streamlit session keys used across the app."""

    defaults = {
        "api_key": "",
        "job_description": "",
        "uploaded_files": [],
        "parsed_candidates": [],
        "processing_error": "",
    }

    for key, value in defaults.items():
        session_state.setdefault(key, value)


def validate_api_key(api_key: str) -> ValidationResult:
    """Validate the Gemini API key input."""

    cleaned_key = api_key.strip()
    if not cleaned_key:
        return ValidationResult(False, "Enter your Gemini API key to continue.")
    if len(cleaned_key) < 20:
        return ValidationResult(False, "The API key looks too short to be valid.")
    return ValidationResult(True, "API key accepted.")


def validate_uploaded_files(uploaded_files: Sequence[object] | None) -> ValidationResult:
    """Validate the uploaded resume list without parsing content yet."""

    if not uploaded_files:
        return ValidationResult(False, "Upload one or more PDF resumes to begin.")
    return ValidationResult(True, f"{len(uploaded_files)} file(s) queued for processing.")


def validate_job_description(job_description: str) -> ValidationResult:
    """Validate the job description text input."""

    if not job_description.strip():
        return ValidationResult(False, "Paste a job description to enable matching.")
    if len(job_description.strip()) < 50:
        return ValidationResult(False, "Provide a more detailed job description for reliable analysis.")
    return ValidationResult(True, "Job description captured.")


def supported_file_types() -> tuple[str, ...]:
    """Return the file types accepted by the UI."""

    return SUPPORTED_RESUME_MIME_TYPES


def format_status_summary(uploaded_files: Iterable[object], job_description: str, api_key: str) -> list[str]:
    """Build a compact status list for the dashboard."""

    statuses = []
    statuses.append(f"Resumes uploaded: {len(list(uploaded_files))}")
    statuses.append(f"Job description length: {len(job_description.strip())} characters")
    statuses.append(f"API key present: {'yes' if api_key.strip() else 'no'}")
    return statuses


def placeholder_processing_message() -> str:
    """Return a friendly message for roadmap steps not yet implemented."""

    return "Processing logic will be added in the next roadmap step."


def raise_not_implemented(feature_name: str) -> None:
    """Raise a consistent placeholder error for future roadmap work."""

    raise NotImplementedError(f"{feature_name} is not implemented yet.")


def _uploaded_file_to_bytes(uploaded_file: Any) -> bytes:
    """Convert a Streamlit uploaded file into raw bytes safely."""

    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()
    if hasattr(uploaded_file, "read"):
        return uploaded_file.read()
    raise TypeError("Unsupported uploaded file object. Expected a file-like object.")


def extract_pdf_text(uploaded_file: Any) -> str:
    """Extract text from a PDF resume using pdfplumber."""

    pdf_bytes = _uploaded_file_to_bytes(uploaded_file)
    pages_text: list[str] = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            cleaned_text = page_text.strip()
            if cleaned_text:
                pages_text.append(cleaned_text)

    extracted_text = "\n\n".join(pages_text).strip()
    if not extracted_text:
        file_name = getattr(uploaded_file, "name", "the uploaded PDF")
        raise ValueError(f"No readable text was found in {file_name}.")

    return extracted_text


def build_resume_prompt(resume_text: str) -> str:
    """Build the Gemini prompt used for structured resume extraction."""

    return (
        "You are a strict resume parsing engine.\n"
        "Extract the candidate information from the resume text below.\n"
        "Return only the fields required by the schema.\n"
        "If a field is missing, use an empty string or an empty list.\n\n"
        f"RESUME TEXT:\n{resume_text}"
    )


def parse_resume_with_ai(
    client: genai.Client,
    resume_text: str,
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> CandidateProfile:
    """Parse a raw resume into a structured candidate profile using Gemini."""

    prompt = build_resume_prompt(resume_text)
    config = types.GenerateContentConfig(
        responseMimeType="application/json",
        responseSchema=CandidateProfile,
        temperature=0,
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    )

    raw_json = getattr(response, "text", None)
    if not raw_json:
        raise ValueError("Gemini did not return a JSON response for the resume.")

    try:
        return CandidateProfile.model_validate_json(raw_json)
    except Exception as exc:  # pragma: no cover - defensive fallback for SDK output
        raise ValueError("Gemini returned data that did not match the resume schema.") from exc
