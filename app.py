import streamlit as st
import utils

# 1. Page Configuration
st.set_page_config(page_title="AI Resume Screening & Ranking", layout="wide")

# 2. Sidebar Layout (API Key ONLY)
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Gemini API Key", type="password", placeholder="Paste your API key here...")

# 3. Main Dashboard Layout
st.title("🤖 AI-Powered Resume Screening & Candidate Ranking")
st.markdown("---")

# Main columns for inputs only
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Job Description")
    jd_input = st.text_area(
        "Paste the target role description",
        placeholder="Add the job requirements, responsibilities, and must-have skills here...",
        height=300,
    )

with col2:
    st.subheader("📂 Resume Intake")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF resumes",
        type=["pdf"],
        accept_multiple_files=True,
    )

st.markdown("---")

# 4. Action Button
if st.button("Analyze Candidates", type="primary", use_container_width=True):
    if not api_key:
        st.error("Please provide your Gemini API key in the sidebar.")
    elif not jd_input.strip():
        st.warning("Please paste a Job Description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        st.success("Inputs verified. Ready for Step 2 analysis.")
