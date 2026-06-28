from __future__ import annotations

import streamlit as st
from google import genai

import utils


st.set_page_config(page_title="AI Resume Screening & Ranking", layout="wide")

utils.initialize_session_state(st.session_state)

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API Key", type="password", placeholder="Paste your API key here...")

st.title("AI-Powered Resume Screening & Candidate Ranking")
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Job Description")
    jd_input = st.text_area(
        "Paste the target role description",
        placeholder="Add the job requirements, responsibilities, and must-have skills here...",
        height=300,
    )

with col2:
    st.subheader("Resume Intake")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF resumes",
        type=["pdf"],
        accept_multiple_files=True,
    )

st.markdown("---")

if st.button("Analyze Candidates", type="primary", use_container_width=True):
    if not api_key:
        st.error("Please provide your Gemini API key in the sidebar.")
    elif not jd_input.strip():
        st.warning("Please paste a Job Description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        try:
            client = genai.Client(api_key=api_key)
            parsed_candidates: list[dict] = []
            parse_errors: list[str] = []

            with st.spinner("Extracting and parsing resumes..."):
                for uploaded_file in uploaded_files:
                    try:
                        resume_text = utils.extract_pdf_text(uploaded_file)
                        candidate_profile = utils.parse_resume_with_ai(client, resume_text)
                        parsed_candidates.append(candidate_profile.model_dump())
                    except Exception as exc:
                        parse_errors.append(f"{getattr(uploaded_file, 'name', 'resume.pdf')}: {exc}")

            st.session_state["parsed_candidates"] = parsed_candidates

            if parsed_candidates and not parse_errors:
                st.success(f"Parsed {len(parsed_candidates)} resume(s) successfully. Ready for Step 3.")
            elif parsed_candidates and parse_errors:
                st.warning(f"Parsed {len(parsed_candidates)} resume(s) with {len(parse_errors)} error(s).")
                for error in parse_errors:
                    st.error(error)
            else:
                st.error("No resumes could be parsed. Please review the uploaded files and try again.")
                for error in parse_errors:
                    st.error(error)
        except Exception as exc:
            st.session_state["parsed_candidates"] = []
            st.error("An unexpected error occurred while processing resumes.")
            st.exception(exc)
