import streamlit as st
from utils import *

def show_upload_section():
    """Display the file upload and settings section"""
    st.markdown("<h2 style='color: var(--accent);'>Upload Your Resume</h2>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drag and drop your resume file here (PDF, TXT, or DOCX)",
        type=["pdf", "txt", "docx"],
        help="Maximum file size: 5MB",
        label_visibility="collapsed"
    )
    
    job_role = st.text_input(
        "Target Job Role (optional - for tailored feedback)",
        placeholder="e.g. Data Scientist, Marketing Manager"
    )
    
    show_preview = st.checkbox("Show extracted text preview", value=False)

    if uploaded_file and show_preview:
        with st.expander("Extracted Text Preview"):
            if st.session_state.extracted_text:
                st.text(st.session_state.extracted_text["text"][:2000] + "...")
            else:
                st.warning("Text not extracted yet. Click 'Analyze My Resume' first.")

    return uploaded_file, job_role

def main():
    init_session_state()
    
    st.title("📄 Resume Analysis")
    
    # Upload and analysis flow
    uploaded_file, job_role = show_upload_section()

    if st.button("Analyze My Resume", type="primary", use_container_width=True):
        if not uploaded_file:
            st.warning("Please upload a resume file first")
            st.stop()

        # Validate and extract text
        with st.spinner("Validating and processing your resume..."):
            extracted_data = extract_text_from_file(uploaded_file)
            if not extracted_data or not extracted_data.get("text"):
                st.error("Failed to extract text from the file")
                st.stop()
            
            st.session_state.extracted_text = extracted_data
            word_count = len(extracted_data["text"].split())
            
            if word_count < 50:
                st.warning(f"The resume seems very short ({word_count} words). Please check if text was properly extracted.")
            elif word_count > 1000:
                st.info(f"Processing large resume ({word_count} words). This may take a moment...")

        # Get AI feedback
        with st.spinner("Analyzing your resume with AI..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                client = get_groq_client()
                if not client:
                    st.error("Failed to connect to AI service. Please try again later.")
                    st.stop()
                
                prompt = generate_prompt(
                    extracted_data["text"], 
                    job_role,
                    st.session_state.domain,
                    st.session_state.language
                )
                
                # Update progress
                for percent in range(0, 101, 10):
                    progress_bar.progress(percent)
                    status_text.text(f"Analysis in progress... {percent}%")
                    time.sleep(0.2)
                
                # Make API call
                messages = [
                    {
                        "role": "system", 
                        "content": "You are an expert resume reviewer with 10+ years in HR and recruiting."
                    },
                    {"role": "user", "content": prompt}
                ]
                
                response = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000
                )
                
                feedback = response.choices[0].message.content
                summary_info = extract_summary_info(feedback)
                st.session_state.feedback_data = summary_info
                st.session_state.analysis_complete = True
                
                # Add to history
                st.session_state.resume_history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "rating": summary_info["rating"],
                    "ats_score": summary_info["ats_score"],
                    "job_suggestion": summary_info["job_suggestion"],
                    "raw_feedback": feedback
                })
                
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                st.success("Analysis Complete!")
                st.balloons()

            except Exception as e:
                progress_bar.empty()
                status_text.error(f"Analysis failed: {str(e)}")
                st.error("Please try again or contact support if the problem persists")
                st.stop()

    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.feedback_data:
        show_analysis_results(st.session_state.feedback_data)

if __name__ == "__main__":
    main()