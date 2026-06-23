"""
Gemini integration, scoring, and export logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


SUPPORTED_RESUME_MIME_TYPES: tuple[str, ...] = ("application/pdf",)


@dataclass(frozen=True)
class ValidationResult:
    """Represents a safe validation outcome for user inputs."""

    is_valid: bool
    message: str = ""


def initialize_session_state(session_state: dict) -> None:
    """Seed Streamlit session keys used across the app.

    The function is intentionally idempotent so it can run on every rerun.
    """

    defaults = {
        "api_key": "",
        "job_description": "",
        "uploaded_files": [],
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
