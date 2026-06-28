from __future__ import annotations

from io import StringIO

import pandas as pd
import streamlit as st

import utils


st.set_page_config(page_title="AI Resume Screening & Ranking", layout="wide")

utils.initialize_session_state(st.session_state)
st.session_state.setdefault("structured_jd", None)
st.session_state.setdefault("ranked_results", [])

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

if st.button("Analyze & Rank Candidates", type="primary", use_container_width=True):
    if not api_key:
        st.error("Please provide your Gemini API key in the sidebar.")
    elif not jd_input.strip():
        st.warning("Please paste a Job Description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        try:
            with st.spinner("Analyzing job description and ranking candidates..."):
                structured_jd = utils.analyze_job_description(jd_input, api_key)

                ranked_candidates: list[dict] = []
                parse_errors: list[str] = []
                for uploaded_file in uploaded_files:
                    try:
                        resume_text = utils.extract_pdf_text(uploaded_file)
                        candidate_profile = utils.parse_resume_with_ai(resume_text, api_key)
                        match_score = utils.calculate_match_score(resume_text, jd_input)
                        candidate_record = candidate_profile.model_dump()
                        candidate_record["filename"] = getattr(uploaded_file, "name", "resume.pdf")
                        candidate_record["match_score"] = match_score
                        ranked_candidates.append(candidate_record)
                    except Exception as file_exc:
                        parse_errors.append(f"{getattr(uploaded_file, 'name', 'resume.pdf')}: {file_exc}")

                ranked_candidates.sort(key=lambda candidate: candidate.get("match_score", 0), reverse=True)

                st.session_state["structured_jd"] = structured_jd.model_dump()
                st.session_state["ranked_results"] = ranked_candidates

            st.success(f"Processed {len(ranked_candidates)} candidate(s) successfully.")
            if parse_errors:
                st.warning(f"{len(parse_errors)} resume(s) could not be processed.")
                for error in parse_errors:
                    st.error(error)
        except Exception as exc:
            st.session_state["structured_jd"] = None
            st.session_state["ranked_results"] = []
            st.error("An unexpected error occurred while processing the candidates.")
            st.exception(exc)


structured_jd = st.session_state.get("structured_jd")
ranked_results = st.session_state.get("ranked_results", [])

if structured_jd:
    with st.expander("Structured Job Description", expanded=True):
        st.write("Experience Required")
        st.write(structured_jd.get("experience_required", ""))
        st.write("Required Skills")
        st.write(structured_jd.get("required_skills", []))
        st.write("Preferred Qualifications")
        st.write(structured_jd.get("preferred_qualifications", []))

if ranked_results:
    leaderboard_rows = []
    for index, candidate in enumerate(ranked_results, start=1):
        leaderboard_rows.append(
            {
                "Rank": index,
                "Candidate Name": candidate.get("name", ""),
                "Email": candidate.get("email", ""),
                "Match Score": candidate.get("match_score", 0),
                "Filename": candidate.get("filename", ""),
            }
        )

    leaderboard_df = pd.DataFrame(leaderboard_rows)

    st.subheader("Candidate Leaderboard")
    st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

    csv_buffer = StringIO()
    leaderboard_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Leaderboard CSV",
        data=csv_buffer.getvalue(),
        file_name="candidate_leaderboard.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.subheader("Candidate Deep Dive")
    for index, candidate in enumerate(ranked_results, start=1):
        title = f"Rank {index}: {candidate.get('name', 'Unknown Candidate')} ({candidate.get('filename', 'resume.pdf')})"
        with st.expander(title):
            st.write("Email")
            st.write(candidate.get("email", ""))
            st.write("Phone")
            st.write(candidate.get("phone", ""))
            st.write("Skills")
            st.write(candidate.get("skills", []))
            st.write("Certifications")
            st.write(candidate.get("certifications", []))
            st.write("Work History")
            st.write(candidate.get("work_experience", []))
            st.write("Projects")
            st.write(candidate.get("projects", []))
