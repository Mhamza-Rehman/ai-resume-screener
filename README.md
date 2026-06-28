# ai-resume-screener
Task 2 of teyzix Program

Features
PDF Multi-Resume Parsing: Fast text extraction using `pdfplumber`

Structured AI Profiling: Uses `gemini-2.5-flash` with strict Pydantic parsing schemas to consistently extract skills, education, experience, and certifications according to the defined schema

Semantic Match Engine: Uses a localized `sentence-transformers` embedding model and cosine similarity to rank candidates accurately without hitches or API rate limits.

HR Dashboard Leaderboard: Interactive data tables displaying ranked candidates with a detailed expander breakdown for deep revieww.
Data Portability: Clean export feature to download the entire structured leaderboard as a CSV file instantly.

---------------------------

## 🛠️ Architecture & Tech Stack
Streamlit
LLM Engine:Google GenAI SDK (`gemini-2.5-flash`)
Data Processing:Pydantic schemas (Structured Outputs), Pandas, NumPy
NLP & Vector Embeddings: Sentence-Transformers (`all-MiniLM-L6-v2`), Scikit-Learn

## 💻 Installation & Setup
Install Dependencies
pip install -r requirements.txt

Use to run app 
python -m streamlit run app.py