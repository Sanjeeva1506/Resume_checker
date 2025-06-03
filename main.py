import streamlit as st
import PyPDF2
import io
import os
import re
import time
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="AI Resume Critiquer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS from external file
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize Groq client
def get_groq_client():
    try:
        return Groq(api_key=os.getenv("GROQ_API_KEY"))
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {str(e)}")
        return None


# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip() if text else None
    except Exception as e:
        st.error(f"Failed to extract text from PDF: {str(e)}")
        return None


# Extract text from uploaded file
def extract_text_from_file(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
        return uploaded_file.read().decode("utf-8").strip()
    except Exception as e:
        st.error(f"Failed to process file: {str(e)}")
        return None


# Generate analysis prompt
def generate_prompt(resume_text, job_role=None):
    base_prompt = """Please analyze this resume and provide honest feedback only for mentioned {role} using EXACTLY the following format:
    if resume not suitable for mentioned {role}, give low rating.

### Key Strengths
- [Your strength 1]
- [Your strength 2]

### Weaknesses
- [Your weakness 1]
- [Your weakness 2]

### Experience Analysis
- [Experience 1]
- [Experience 2]

### Job Targeting
- [Tailored advice for {role}]

### ATS Optimization
- [Tip 1]
- [Tip 2]

### Content Clarity
- [Improvement 1]
- [Improvement 2]

### Rating: X/10

### Suggested Role: [Role]

Resume content:
{resume}
"""
    return base_prompt.format(
        role=job_role if job_role else "general job applications",
        resume=resume_text
    )


# Improved extraction of summary info from AI feedback
def extract_summary_info(feedback):
    # Safely extract rating
    rating_match = re.search(r'Rating:\s*(\d+\.?\d*)\s*/\s*10', feedback, re.IGNORECASE)
    rating = rating_match.group(1) if rating_match else "N/A"

    # Safely extract job suggestion
    job_match = re.search(
        r'(?:Suggested role|Recommended position|Target role):\s*(.+?)(?=\n\d+\.|\n[A-Z]|\n\n|$)',
        feedback,
        re.IGNORECASE | re.DOTALL
    )
    job_suggestion = job_match.group(1).strip() if job_match else "Not specified"

    # Safely extract strengths
    strength_section = ""
    # Look for any variation of the strengths heading
    strength_start = re.search(
        r'(?:###\s*Key\s*Strengths|###\s*Strengths?|\*\*Strengths?\*\*:?|Strengths?\s*:\s*)',
        feedback,
        re.IGNORECASE
    )
    if strength_start:
        start_idx = strength_start.end()
        # Find end of section (next heading or EOF)
        end_idx = re.search(
            r'\n\s*(?:###|\*\*|Rating:|Suggested role:|Weaknesses?)',
            feedback[start_idx:], re.IGNORECASE
        )
        strength_section = feedback[start_idx:] if not end_idx else feedback[start_idx:start_idx + end_idx.start()]

    strengths = []
    if strength_section:
        lines = strength_section.strip().split('\n')
        for line in lines:
            cleaned = line.strip('-•* ').strip()
            if cleaned and not cleaned.lower().startswith(('weakness', 'rating')):
                strengths.append(cleaned)
            if len(strengths) >= 5:
                break

    return {
        "rating": rating,
        "job_suggestion": job_suggestion,
        "strengths": strengths
    }


# Main app function
def main():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("AI Resume Critiquer")
        st.markdown("""
        <p style="font-size: 18px;">
            Get expert-level feedback on your resume instantly!  
            Our AI analyzes your resume for ATS compatibility, content impact, and job-specific optimizations.
        </p>
        """, unsafe_allow_html=True)
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/3132/3132693.png",  width=150)

    # Sidebar
    with st.sidebar:
        st.markdown("<h2 style='color: var(--accent);'>Settings</h2>", unsafe_allow_html=True)
        st.markdown("Customize your analysis:")
        model_choice = st.selectbox(
            "AI Model",
            ["llama3-70b-8192", "llama3-8b-8192"],
            index=0,
            help="70b is more thorough, 8b is faster"
        )
        temperature = st.slider(
            "Analysis Style",
            0.0, 1.0, 0.7,
            help="Lower = more factual, Higher = more creative"
        )
        st.markdown("---")
        st.markdown("<h4 style='color: var(--accent);'>How it works:</h4>", unsafe_allow_html=True)
        st.markdown("1. Upload your resume (PDF or TXT)")
        st.markdown("2. Enter target job (optional)")
        st.markdown("3. Get instant expert feedback")
        st.markdown("---")
        

    # Upload section
    st.markdown("<h2 style='color: var(--accent);'>Upload Your Resume</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag and drop your resume file here (PDF or TXT)",
        type=["pdf", "txt"],
        help="Maximum file size: 5MB",
        label_visibility="collapsed"
    )
    job_role = st.text_input(
        "Target Job Role (optional - for tailored feedback)",
        placeholder="e.g. Data Scientist, Marketing Manager"
    )

    if st.button("Analyze My Resume", type="primary", use_container_width=True):
        if not uploaded_file:
            st.warning("Please upload a resume file first")
            return

        with st.spinner("Analyzing your resume... This may take a moment"):
            progress_bar = st.empty()
            progress_bar.markdown("""
            <div class="progress-bar">
                <div class="progress-fill" id="progress"></div>
            </div>
            """, unsafe_allow_html=True)

            for percent in range(0, 101, 5):
                time.sleep(0.05)
                progress_bar.markdown(f"""
                <div class="progress-bar">
                    <div class="progress-fill" style="width:{percent}%"></div>
                </div>
                """, unsafe_allow_html=True)

            try:
                file_content = extract_text_from_file(uploaded_file)
                if not file_content:
                    st.error("Could not extract meaningful text from the file")
                    return

                if len(file_content.split()) < 50:
                    st.warning("The resume seems very short. Please check if text was properly extracted.")

                client = get_groq_client()
                if not client:
                    return

                prompt = generate_prompt(file_content, job_role)
                response = client.chat.completions.create(
                    model=model_choice,
                    messages=[
                        {"role": "system", "content": "You are an expert resume reviewer with 10+ years in HR and recruiting. Provide detailed, actionable feedback."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=1500,
                    top_p=0.9
                )

                progress_bar.empty()
                st.success("Analysis Complete!")
                st.balloons()

                feedback = response.choices[0].message.content
                summary_info = extract_summary_info(feedback)

                # Display results
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.markdown("<h3 style='color: var(--accent);'>Quick Summary</h3>", unsafe_allow_html=True)

                    # Rating card
                    st.markdown(f"""
                    <div class="feedback-card">
                        <h4 style="text-align: center; color: var(--accent);">Overall Rating</h4>
                        <div class="rating-circle">{summary_info['rating']}/10</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="feedback-card">
                        <h4 style="color: var(--accent);">Suggested Role</h4>
                        <p style="font-size: 16px;">{summary_info['job_suggestion']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if summary_info['strengths']:
                        strengths_html = "<ul>" + "".join([f"<li>{s}</li>" for s in summary_info['strengths']]) + "</ul>"
                        st.markdown(f"""
                        <div class="feedback-card">
                            <h4 style="color: var(--accent);">Key Strengths</h4>
                            {strengths_html}
                        </div>
                        """, unsafe_allow_html=True)

                with col2:
                    st.markdown("<h3 style='color: var(--accent);'>Detailed Feedback</h3>", unsafe_allow_html=True)
                    st.markdown(feedback)

                # Action buttons
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button(
                        label="Download Feedback",
                        data=feedback,
                        file_name="resume_feedback.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                with col2:
                    if st.button("Analyze Another Resume", use_container_width=True):
                        st.experimental_rerun()
                with col3:
                    if st.button("Email Me This Feedback", use_container_width=True):
                        st.info("This feature would connect to your email service in a production app")

            except Exception as e:
                progress_bar.empty()
                st.error(f"An error occurred during analysis: {str(e)}")
                st.error("Please try again or contact support if the problem persists")


if __name__ == "__main__":
    main()