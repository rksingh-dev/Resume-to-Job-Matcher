import os

# Add at the beginning of resume-to-job-mathcher.py
PORT = int(os.environ.get("PORT", 8501))
import streamlit as st
import spacy
import os
from docx import Document
import fitz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Load models
@st.cache_resource
def load_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    jobbert_pipeline = pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple"
    )
    return model, jobbert_pipeline

model, jobbert_pipeline = load_models()

# Predefined skills list
PREDEFINED_SKILLS = set([
    "Python", "Java", "C++", "JavaScript", "SQL", "Machine Learning", "Deep Learning",
    "Artificial Intelligence", "Data Science", "NLP", "TensorFlow", "PyTorch", "Keras",
    "Flask", "Django", "FastAPI", "React", "Angular", "Vue.js", "Node.js",
    "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "Git", "DevOps",
    "Cybersecurity", "Blockchain", "Big Data", "Linux", "Unix", "Embedded Systems",
    "Agile", "Scrum", "JIRA", "Power BI", "Tableau", "Software Testing",
    "Android Development", "iOS Development", "React Native", "Flutter",
    "Natural Language Processing", "Computer Vision", "MLOps", "ETL", "Data Engineering"
])

def extract_text(file):
    try:
        ext = os.path.splitext(file.name)[-1].lower()
        text = ""
        
        if ext == ".pdf":
            doc = fitz.open(stream=file.read(), filetype="pdf")
            for page in doc:
                text += page.get_text("text")
        
        elif ext == ".docx":
            doc = Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
        
        elif ext == ".txt":
            text = file.read().decode("utf-8")
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        return text.strip()
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return ""

def extract_skills(text):
    """Extract skills from text by matching against predefined skills."""
    found_skills = set()
    text_lower = text.lower()
    
    for skill in PREDEFINED_SKILLS:
        if skill.lower() in text_lower:
            found_skills.add(skill)
    
    return found_skills

def extract_domain_terms(text):
    """Extract domain-specific terms using NER."""
    try:
        entities = jobbert_pipeline(text)
        domain_terms = {
            entity["word"] 
            for entity in entities 
            if entity["entity"].startswith("B-") and 
               entity["score"] > 0.8  # Only include high-confidence predictions
        }
        return domain_terms
    except Exception as e:
        st.warning(f"Error in domain term extraction: {str(e)}")
        return set()

def compute_weighted_similarity(resume_text, job_desc_text):
    """Compute weighted similarity between resume and job description."""
    try:
        # Extract skills and domain terms
        resume_skills = extract_skills(resume_text) | extract_domain_terms(resume_text)
        job_skills = extract_skills(job_desc_text) | extract_domain_terms(job_desc_text)
        
        # Calculate skill match score
        skill_match_score = len(resume_skills.intersection(job_skills)) / max(len(job_skills), 1)
        
        # Calculate embedding similarity
        embedding_similarity = cosine_similarity(
            [model.encode(resume_text)], 
            [model.encode(job_desc_text)]
        )[0][0]
        
        # Calculate weighted score
        weighted_score = (0.7 * embedding_similarity) + (0.3 * skill_match_score)
        
        return {
            'weighted_score': weighted_score,
            'embedding_similarity': embedding_similarity,
            'skill_match_score': skill_match_score,
            'matching_skills': resume_skills.intersection(job_skills),
            'missing_skills': job_skills - resume_skills
        }
    except Exception as e:
        st.error(f"Error computing similarity: {str(e)}")
        return None

def main():
    st.title("Resume to Job Matcher")
    st.write("Upload your resume and a job description to find out how well they match!")

    resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
    job_desc_file = st.file_uploader("Upload Job Description", type=["pdf", "docx", "txt"])

    if resume_file and job_desc_file:
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Extract text from files
                resume_text = extract_text(resume_file)
                job_desc_text = extract_text(job_desc_file)

                if resume_text and job_desc_text:
                    # Compute similarity
                    results = compute_weighted_similarity(resume_text, job_desc_text)
                    
                    if results:
                        # Display results
                        st.subheader("Results")
                        st.write(f"Overall Match Score: {results['weighted_score']:.2%}")
                        st.write(f"Text Similarity: {results['embedding_similarity']:.2%}")
                        st.write(f"Skills Match: {results['skill_match_score']:.2%}")
                        
                        st.subheader("Matching Skills")
                        st.write(", ".join(sorted(results['matching_skills'])) or "No matching skills found")
                        
                        st.subheader("Missing Skills")
                        st.write(", ".join(sorted(results['missing_skills'])) or "No missing skills")

@st.cache_resource
def health_check():
    return {"status": "healthy"}

if "health" in st.session_state:
    health_check()

if __name__ == "__main__":
    main()