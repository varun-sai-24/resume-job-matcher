import streamlit as st
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Text preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return set(text.split())

# PDF extraction function
def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()
    return text

# Streamlit UI
st.title("📄 Resume vs Job Description Matcher")

st.markdown("Upload your **Resume PDF** and a **Job Description** to get a match score and keyword analysis.")

resume_pdf = st.file_uploader("Upload Resume PDF", type=["pdf"])
job_description = st.text_area("Paste Job Description Here", height=200)

if st.button("Match"):
    if resume_pdf is not None and job_description.strip() != "":
        resume_text = extract_text_from_pdf(resume_pdf)
        job_text = job_description

        # TF-IDF Match Score
        documents = [resume_text, job_text]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        # Keyword Comparison
        resume_words = preprocess(resume_text)
        job_words = preprocess(job_text)

        matched_keywords = resume_words.intersection(job_words)
        missing_keywords = job_words.difference(resume_words)

        # Results
        st.success(f"🔎 Match Score: **{similarity_score * 100:.2f}%**")
        
        st.markdown("### ✅ Matched Keywords")
        st.write(", ".join(sorted(matched_keywords)) if matched_keywords else "No keywords matched.")

        st.markdown("### ❌ Missing Keywords")
        st.write(", ".join(sorted(missing_keywords)) if missing_keywords else "None! Great match.")

    else:
        st.warning("Please upload a resume and paste a job description.")
