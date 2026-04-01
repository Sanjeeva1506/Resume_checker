import streamlit as st
from utils import *

def handle_multi_file_upload():
    """Handle multiple resume uploads"""
    st.markdown("<h3 style='color: var(--accent);'>Upload Multiple Resumes</h3>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Select multiple resume files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Maximum 20 files, 5MB each",
        key="multi_file_uploader"
    )
    
    if uploaded_files and len(uploaded_files) > 20:
        st.error("Maximum 20 files allowed. Please select fewer files.")
        return
    
    if st.button("Process Uploaded Resumes", key="process_multiple"):
        with st.spinner(f"Processing {len(uploaded_files)} resumes..."):
            successful_uploads = 0
            for file in uploaded_files:
                try:
                    extracted_data = extract_text_from_file(file)
                    if extracted_data and extracted_data.get("text"):
                        file_id = base64.b64encode(file.name.encode()).decode()[:20]
                        
                        st.session_state.uploaded_resumes.append({
                            "id": file_id,
                            "name": file.name,
                            "size": file.size,
                            "type": file.type,
                            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "extracted_data": extracted_data,
                            "analysis": None,
                            "tags": [],
                            "metadata": {
                                "page_count": extracted_data.get("page_count", 1),
                                "word_count": len(extracted_data["text"].split()),
                                "sections": extracted_data.get("sections", [])
                            }
                        })
                        successful_uploads += 1
                except Exception as e:
                    logger.error(f"Failed to process {file.name}: {str(e)}")
            
            if successful_uploads > 0:
                st.success(f"Successfully processed {successful_uploads} resumes!")
            else:
                st.error("No resumes were successfully processed")

def show_resume_dashboard():
    """Display dashboard for managing multiple resumes"""
    st.markdown("<h2 style='color: var(--accent);'>Resume Management Dashboard</h2>", unsafe_allow_html=True)
    
    if not st.session_state.uploaded_resumes:
        st.info("No resumes uploaded yet. Use the upload section to add resumes.")
        return
    
    # Sorting controls
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.session_state.sort_criteria = st.selectbox(
            "Sort by",
            ["upload_time", "name", "size", "page_count", "word_count"],
            format_func=lambda x: {
                "upload_time": "Upload Time",
                "name": "File Name",
                "size": "File Size",
                "page_count": "Page Count",
                "word_count": "Word Count"
            }[x]
        )
    with col2:
        st.session_state.sort_order = st.selectbox(
            "Order",
            ["desc", "asc"],
            format_func=lambda x: "Descending" if x == "desc" else "Ascending"
        )
    with col3:
        if st.button("Refresh View"):
            st.rerun()
    
    # Apply sorting
    reverse_order = st.session_state.sort_order == "desc"
    sorted_resumes = sorted(
        st.session_state.uploaded_resumes,
        key=lambda x: x.get("metadata", {}).get(st.session_state.sort_criteria, x.get(st.session_state.sort_criteria, "")),
        reverse=reverse_order
    )
    
    # Display resume cards
    for idx, resume in enumerate(sorted_resumes):
        with st.expander(f"{resume['name']} - {resume['upload_time']}", expanded=False):
            cols = st.columns([3, 1, 1])
            with cols[0]:
                st.markdown(f"""
                **Metadata:**
                - Pages: {resume['metadata']['page_count']}
                - Words: {resume['metadata']['word_count']}
                - Sections: {', '.join(resume['metadata']['sections'][:3])}{'...' if len(resume['metadata']['sections']) > 3 else ''}
                """)
                
                # Tags management
                tag_key = f"tag_input_{idx}_{resume['id']}"
                new_tag = st.text_input(
                    "Add tag",
                    key=tag_key,
                    placeholder="e.g. 'Engineer', 'Entry-level'"
                )
                if new_tag and st.button(f"Add Tag {idx}", key=f"add_tag_{idx}_{resume['id']}"):
                    if new_tag not in st.session_state.resume_tags[resume['id']]:
                        st.session_state.resume_tags[resume['id']].append(new_tag)
                        st.rerun()
                
                if st.session_state.resume_tags.get(resume.get('id', ''), []):
                    st.markdown("**Tags:** " + ", ".join([
                        f"`{tag}`" for tag in st.session_state.resume_tags.get(resume.get('id', ''), [])
                    ]))
            
            with cols[1]:
                if st.button("Analyze", key=f"analyze_{idx}_{resume['id']}"):
                    st.session_state.current_resume_index = next(
                        i for i, r in enumerate(st.session_state.uploaded_resumes) 
                        if r['id'] == resume['id']
                    )
                    st.session_state.extracted_text = resume['extracted_data']
                    st.switch_page("pages/1_📄_Analyze.py")
            
            with cols[2]:
                if st.button("Delete", key=f"delete_{idx}_{resume['id']}"):
                    st.session_state.uploaded_resumes = [
                        r for r in st.session_state.uploaded_resumes 
                        if r['id'] != resume['id']
                    ]
                    st.rerun()

    # Bulk actions
    st.markdown("---")
    st.markdown("<h4>Bulk Actions</h4>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Analyze All Resumes", help="This may take some time"):
            with st.spinner("Analyzing all resumes..."):
                for resume in st.session_state.uploaded_resumes:
                    if not resume['analysis']:
                        try:
                            client = get_groq_client()
                            prompt = generate_prompt(
                                resume['extracted_data']["text"],
                                domain=st.session_state.domain,
                                language=st.session_state.language
                            )
                            
                            response = client.chat.completions.create(
                                model="llama3-70b-8192",
                                messages=[
                                    {"role": "system", "content": "You are an expert resume reviewer."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.7,
                                max_tokens=2000
                            )
                            
                            feedback = response.choices[0].message.content
                            summary_info = extract_summary_info(feedback)
                            resume['analysis'] = summary_info
                            
                        except Exception as e:
                            logger.error(f"Failed to analyze {resume['name']}: {str(e)}")
                
                st.success("Bulk analysis completed!")
    with col2:
        if st.button("Clear All Resumes", type="secondary"):
            st.session_state.uploaded_resumes = []
            st.session_state.resume_tags = defaultdict(list)
            st.rerun()

def main():
    init_session_state()
    st.title("📂 Resume Manager")
    
    handle_multi_file_upload()
    st.markdown("---")
    show_resume_dashboard()

if __name__ == "__main__":
    main()