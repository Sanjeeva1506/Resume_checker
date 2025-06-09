import streamlit as st
import PyPDF2
import io
import os
import re
import time
import json
import requests
from groq import Groq
from dotenv import load_dotenv
from typing import Tuple, Optional, Dict, List
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from docx import Document
import pandas as pd
import plotly.express as px
from datetime import datetime
from deep_translator import GoogleTranslator
from langdetect import detect
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from typing import List, Dict, Any
import base64
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="AI Resume Critiquer Pro",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS from external file
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Add new CSS for improvements
st.markdown("""
<style>
/* Section highlighting */
.section-present {
    background-color: rgba(46, 125, 50, 0.1);
    padding: 8px;
    border-left: 3px solid #2e7d32;
    margin-bottom: 10px;
}

.section-missing {
    background-color: rgba(198, 40, 40, 0.1);
    padding: 8px;
    border-left: 3px solid #c62828;
    margin-bottom: 10px;
}

/* Keyword highlighting */
.keyword-match {
    background-color: #fff9c4;
    padding: 2px 4px;
    border-radius: 3px;
    color: #000 !important;
}

/* Mobile responsive adjustments */
@media (max-width: 768px) {
    .feedback-card {
        margin-bottom: 15px;
    }
    
    .rating-circle {
        width: 80px;
        height: 80px;
        font-size: 24px;
    }
    
    .stButton>button {
        padding: 8px 16px;
        font-size: 14px;
    }
}

/* Loading spinner */
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
.loading-spinner {
    border: 4px solid rgba(0, 0, 0, 0.1);
    border-radius: 50%;
    border-top: 4px solid var(--accent);
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
    margin: 20px auto;
}

/* ATS score meter */
.ats-meter {
    height: 20px;
    background: #4b5563;
    border-radius: 10px;
    margin: 10px 0;
    overflow: hidden;
}

.ats-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 10px;
    transition: width 0.5s ease-in-out;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    background: var(--card) !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    margin: 0 !important;
}

.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
}

/* Job card styling */
.job-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
    transition: box-shadow 0.3s;
}

.job-card:hover {
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.job-match-score {
    font-weight: bold;
    color: var(--accent);
}

/* Soft skills tags */
.skill-tag {
    display: inline-block;
    background-color: #e3f2fd;
    color: #1976d2;
    padding: 3px 8px;
    border-radius: 12px;
    margin: 2px;
    font-size: 12px;
}

/* Bias indicator */
.bias-indicator {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 12px;
    margin-left: 5px;
}

.bias-low {
    background-color: #e8f5e9;
    color: #2e7d32;
}

.bias-medium {
    background-color: #fff8e1;
    color: #ff8f00;
}

.bias-high {
    background-color: #ffebee;
    color: #c62828;
}

/* Template preview */
.template-preview {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 10px;
    margin: 10px 0;
    height: 300px;
    overflow-y: auto;
}

/* Language selector */
.language-selector {
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = None
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = None
if 'user_feedback' not in st.session_state:
    st.session_state.user_feedback = {}
if 'job_matches' not in st.session_state:
    st.session_state.job_matches = []
if 'selected_template' not in st.session_state:
    st.session_state.selected_template = None
if 'resume_history' not in st.session_state:
    st.session_state.resume_history = []
if 'language' not in st.session_state:
    st.session_state.language = "en"
if 'domain' not in st.session_state:
    st.session_state.domain = "general"

if 'uploaded_resumes' not in st.session_state:
    st.session_state.uploaded_resumes = []  # Stores all uploaded resumes
if 'current_resume_index' not in st.session_state:
    st.session_state.current_resume_index = 0
if 'sort_criteria' not in st.session_state:
    st.session_state.sort_criteria = "upload_time"
if 'sort_order' not in st.session_state:
    st.session_state.sort_order = "desc"
if 'resume_tags' not in st.session_state:
    st.session_state.resume_tags = defaultdict(list)  # For categorizing resumes

# Error handling decorator
def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PyPDF2.PdfReadError as e:
            logger.error(f"PDF reading error: {str(e)}")
            st.error("The PDF file appears to be corrupted. Please try another file.")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            st.error(f"An unexpected error occurred: {str(e)}")
            st.info("Please try again or contact support if the problem persists")
        return None
    return wrapper

# Initialize Groq client with enhanced error handling
@handle_errors
def get_groq_client(max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            if not api_key.startswith("gsk_") or len(api_key) != 56:
                raise ValueError("Invalid Groq API key format")
            return Groq(api_key=api_key)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt  # Exponential backoff
            logger.warning(f"Retry {attempt + 1} after {wait_time} seconds...")
            time.sleep(wait_time)
    return None

# New function to handle multiple file uploads
def handle_multi_file_upload():
    """Handle multiple resume uploads and initial processing"""
    st.markdown("<h3 style='color: var(--accent);'>Upload Multiple Resumes</h3>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Select multiple resume files (PDF, TXT, or DOCX)",
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
                        # Generate a unique ID for each resume
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
                if successful_uploads < len(uploaded_files):
                    st.warning(f"Failed to process {len(uploaded_files) - successful_uploads} files")
            else:
                st.error("No resumes were successfully processed")

# New function for resume management dashboard
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
                
                # Tags management - using both index and ID for unique key
                tag_key = f"tag_input_{idx}_{resume['id']}"
                new_tag = st.text_input(
                    "Add tag",
                    key=tag_key,
                    placeholder="e.g. 'Engineer', 'Entry-level'"
                )
                if new_tag and st.button(f"Add Tag {idx}", key=f"add_tag_{idx}_{resume['id']}"):
                    if new_tag not in st.session_state.resume_tags[resume['id']]:
                        st.session_state.resume_tags[resume['id']].append(new_tag)
                        # Clear the input field by removing the key from session state
                        if tag_key in st.session_state:
                            del st.session_state[tag_key]
                        st.rerun()
                
                if st.session_state.resume_tags[resume['id']]:
                    st.markdown("**Tags:** " + ", ".join([
                        f"`{tag}`" for tag in st.session_state.resume_tags[resume['id']]
                    ]))
            
            with cols[1]:
                if st.button("Analyze", key=f"analyze_{idx}_{resume['id']}"):
                    st.session_state.current_resume_index = next(
                        i for i, r in enumerate(st.session_state.uploaded_resumes) 
                        if r['id'] == resume['id']
                    )
                    st.session_state.extracted_text = resume['extracted_data']
                    st.rerun()
            
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

# New function for candidate comparison
def show_candidate_comparison():
    """Display comparison view for multiple resumes with comprehensive error handling"""
    
    # Initial checks with proper error handling
    try:
        if not hasattr(st.session_state, 'uploaded_resumes') or len(st.session_state.get('uploaded_resumes', [])) < 2:
            st.info("Upload at least 2 resumes to enable comparison")
            return
    except Exception as e:
        st.error(f"Error checking resume count: {str(e)}")
        return

    st.markdown("<h2 style='color: var(--accent);'>Candidate Comparison</h2>", unsafe_allow_html=True)
    
    # Safely get tags with comprehensive None checking
    all_tags = []
    try:
        if hasattr(st.session_state, 'resume_tags'):
            all_tags = sorted({tag for tags in st.session_state.resume_tags.values() 
                             for tag in (tags if isinstance(tags, list) else [])})
    except Exception:
        all_tags = []

    selected_tags = st.multiselect(
        "Filter by tags (optional)",
        all_tags,
        key="tag_filter"
    )

    # Safely build filtered resumes list
    filtered_resumes = []
    try:
        for r in st.session_state.get('uploaded_resumes', []):
            try:
                if not isinstance(r, dict):
                    continue
                    
                resume_id = r.get('id', '')
                resume_tags = st.session_state.resume_tags.get(resume_id, []) if hasattr(st.session_state, 'resume_tags') else []
                
                if not selected_tags or any(tag in resume_tags for tag in selected_tags):
                    filtered_resumes.append(r)
            except Exception:
                continue
    except Exception as e:
        st.error(f"Error filtering resumes: {str(e)}")
        filtered_resumes = []

    if not filtered_resumes:
        st.warning("No resumes available for comparison")
        return

    # Safely get resume names
    resume_options = []
    for r in filtered_resumes:
        try:
            name = r.get('name', 'Unnamed Resume')
            if not isinstance(name, str):
                name = 'Unnamed Resume'
            resume_options.append(name)
        except Exception:
            resume_options.append('Unnamed Resume')

    selected_resumes = st.multiselect(
        "Select resumes to compare (2-5)",
        resume_options,
        default=resume_options[:min(5, len(resume_options))],
        key="compare_select"
    )

    if len(selected_resumes) < 2:
        st.warning("Please select at least 2 resumes to compare")
        return

    # Build selected_data with comprehensive error handling
    selected_data = []
    for r in filtered_resumes:
        try:
            if not isinstance(r, dict):
                continue
                
            name = r.get('name', '')
            if not isinstance(name, str):
                continue
                
            if name in selected_resumes:
                selected_data.append(r)
        except Exception:
            continue

    def validate_resume(resume):
        required = ['name', 'metadata']
        return all(key in resume for key in required)

    valid_resumes = [r for r in selected_data if validate_resume(r)] if selected_data else []

    # Build comparison data with atomic error handling
    comparison_data = []
    for resume in selected_data:
        try:
            # Safely get all components with nested protection
            resume_dict = resume if isinstance(resume, dict) else {}
            
            # Get analysis data with verification
            analysis = resume_dict.get('analysis', {}) if isinstance(resume_dict.get('analysis'), dict) else {}
            
            # Get metadata with verification
            metadata = resume_dict.get('metadata', {}) if isinstance(resume_dict.get('metadata'), dict) else {
                'word_count': 0,
                'page_count': 0,
                'sections': []
            }

            # Get rating with verification
            rating = 0
            if analysis and 'rating' in analysis and analysis['rating'] not in [None, 'N/A', '']:
                try:
                    rating = float(analysis['rating'])
                except (ValueError, TypeError):
                    rating = 0

            # Get ATS score with verification
            ats_score = 0
            if analysis and 'ats_score' in analysis and analysis['ats_score'] not in [None, 'N/A', '']:
                try:
                    ats_score = float(analysis['ats_score'])
                except (ValueError, TypeError):
                    ats_score = 0

            # Get timestamp with verification
            last_analyzed = 'Never'
            if analysis and 'timestamp' in analysis and analysis['timestamp'] not in [None, '']:
                last_analyzed = analysis['timestamp']
            elif analysis and 'raw_feedback' in analysis and isinstance(analysis['raw_feedback'], str):
                # Try to extract timestamp from raw feedback
                try:
                    time_match = re.search(r'analyzed on (.*?)\n', analysis['raw_feedback'])
                    if time_match:
                        last_analyzed = time_match.group(1)
                except:
                    pass        
            
            # Get resume ID safely
            resume_id = resume.get('id', '')
            
            # Build entry with verified data
            entry = {
                "Name": str(resume.get('name', 'Unnamed Resume')),
                "Word Count": int(metadata.get('word_count', 0)),
                "Page Count": int(metadata.get('page_count', 0)),
                "Sections": len(metadata.get('sections', [])) if isinstance(metadata.get('sections'), list) else 0,
                "Rating": float(rating) if rating != "N/A" else 0,
                "ATS Score": float(ats_score) if ats_score != "N/A" else 0,
                "Last Analyzed": last_analyzed,
                "Tags": ", ".join(map(str, st.session_state.resume_tags.get(resume_id, []))) if hasattr(st.session_state, 'resume_tags') else "None"
            }
            comparison_data.append(entry)

        except Exception as e:
            logger.error(f"Skipping resume {resume.get('name', 'unnamed')} due to: {str(e)}")
            logger.debug(f"Problematic resume data: {resume}")
            continue

    # Display results with error boundary
    try:
        if not comparison_data:
            st.warning("No valid resume data to compare")
            return
            
        df = pd.DataFrame(comparison_data)
        
        # Add analysis status column
        df['Status'] = df.apply(lambda row: 
            "Analyzed" if row['Last Analyzed'] != 'Never' else "Not Analyzed", 
            axis=1)

        # Configure columns with safe defaults
        column_config = {
            "Rating": st.column_config.ProgressColumn(
                "Rating",
                help="Overall resume rating (0-10)",
                format="%.1f",
                min_value=0,
                max_value=10
            ),
            "ATS Score": st.column_config.ProgressColumn(
                "ATS Score",
                help="ATS compatibility score (0-100)",
                format="%.1f",
                min_value=0,
                max_value=100
            ),
            "Last Analyzed": st.column_config.DatetimeColumn(
                "Last Analyzed",
                help="When this resume was last analyzed"
            ),
            "Status": st.column_config.TextColumn(
                "Analysis Status",
                help="Whether this resume has been analyzed"
            ),
            "Tags": st.column_config.TextColumn(
                "Tags",
                help="User-assigned tags for categorization"
            )
        }
        
        st.markdown("### Comparison Table")
        st.dataframe(
            df.set_index("Name"),
            use_container_width=True,
            column_config=column_config
        )
        
        # Show analysis buttons for unanalyzed resumes
        if 'Never' in df['Last Analyzed'].values:
            st.markdown("### Analyze Resumes")
            unanalyzed_names = df[df['Last Analyzed'] == 'Never'].index.tolist()
            
            for name in unanalyzed_names:
                if st.button(f"Analyze {name}"):
                    try:
                        resume = next(r for r in selected_data if r.get('name') == name)
                        st.session_state.current_resume_index = next(
                            i for i, r in enumerate(st.session_state.uploaded_resumes)
                            if r.get('id') == resume.get('id')
                        )
                        st.session_state.extracted_text = resume.get('extracted_data')
                        st.rerun()
                    except Exception as e:
                        st.error(f"Couldn't load resume: {str(e)}")

    except Exception as e:
        st.error(f"Error displaying comparison results: {str(e)}")
    
    # Add download button for comparison data
    if not df.empty:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Comparison Data",
            data=csv,
            file_name="resume_comparison.csv",
            mime="text/csv"
        )
    
    # Visualizations with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Ratings & ATS", "📈 Word Count", "📑 Sections", "📌 Detailed View"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                df,
                x="Name",
                y="Rating",
                title="Resume Ratings Comparison",
                color="Rating",
                color_continuous_scale="Teal",
                text="Rating"
            )
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                df,
                x="Name",
                y="ATS Score",
                title="ATS Scores Comparison",
                color="ATS Score",
                color_continuous_scale="Blues",
                text="ATS Score"
            )
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        # Combined radar chart
        fig = px.line_polar(
            df, 
            r=["Rating", "ATS Score"],
            theta=["Rating", "ATS Score"],
            line_close=True,
            title="Combined Metrics Radar Chart"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                df,
                names="Name",
                values="Word Count",
                title="Word Count Distribution",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                df,
                x="Name",
                y="Word Count",
                title="Word Count Comparison",
                color="Word Count",
                text="Word Count"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Create section comparison heatmap
        all_sections = set()
        for resume in selected_data:
            try:
                all_sections.update(resume.get('metadata', {}).get('sections', []))
            except:
                continue
        
        section_data = []
        for section in sorted(all_sections):
            section_entry = {"Section": section}
            for resume in selected_data:
                section_entry[resume['name']] = 1 if section in resume.get('metadata', {}).get('sections', []) else 0
            section_data.append(section_entry)
        
        section_df = pd.DataFrame(section_data).set_index("Section")
        
        # Display as heatmap
        fig = px.imshow(
            section_df,
            labels=dict(x="Resume", y="Section", color="Present"),
            title="Section Presence Heatmap",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Detailed view with expandable sections
        for resume in selected_data:
            rating = resume.get('analysis', {}).get('rating', 'Not Analyzed') if resume.get('analysis') else 'Not Analyzed'
            with st.expander(f"📄 {resume['name']} - Rating: {rating}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Basic Info**")
                    st.markdown(f"""
                    - **Uploaded:** {resume.get('upload_time', 'Unknown')}
                    - **Pages:** {resume.get('metadata', {}).get('page_count', 0)}
                    - **Words:** {resume.get('metadata', {}).get('word_count', 0)}
                    - **Sections:** {', '.join(resume.get('metadata', {}).get('sections', [])[:5])}{'...' if len(resume.get('metadata', {}).get('sections', [])) > 5 else ''}
                    """)
                    
                    if st.session_state.resume_tags.get(resume.get('id', ''), []):
                        st.markdown("**Tags:** " + ", ".join([
                            f"`{tag}`" for tag in st.session_state.resume_tags.get(resume.get('id', ''), [])
                        ]))
                
                with col2:
                    if resume.get('analysis'):
                        st.markdown("**Analysis Results**")
                        st.markdown(f"""
                        - **Rating:** {resume['analysis'].get('rating', 'N/A')}
                        - **ATS Score:** {resume['analysis'].get('ats_score', 'N/A')}
                        - **Suggested Role:** {resume['analysis'].get('job_suggestion', 'N/A')}
                        """)
                
                if resume.get('analysis'):
                    show_feedback = st.toggle("🔍 Show Full Analysis", key=f"toggle_{resume.get('id', '')}")
                    if show_feedback:
                        st.markdown(resume['analysis'].get('raw_feedback', 'No analysis available'))
                
                if st.button(f"Re-analyze {resume['name']}", key=f"reanalyze_{resume.get('id', '')}"):
                    with st.spinner(f"Analyzing {resume['name']}..."):
                        try:
                            client = get_groq_client()
                            prompt = generate_prompt(
                                resume.get('extracted_data', {}).get("text", ""),
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
                            summary_info['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M")
                            resume['analysis'] = summary_info
                            st.success("Re-analysis complete!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to re-analyze: {str(e)}")

# Enhanced file validation with content checking
def validate_file(uploaded_file) -> Tuple[bool, str]:
    """Validate the uploaded file"""
    if uploaded_file.size > 5 * 1024 * 1024:  # 5MB
        return False, "File size exceeds 5MB limit"
    if uploaded_file.type not in ["application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        return False, "Only PDF, TXT, and DOCX files are supported"
    
    # Validate file content
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)  # Reset file pointer
    
    if uploaded_file.type == "application/pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            if len(pdf_reader.pages) > 15:
                return False, "PDF has too many pages (15+). Please upload a shorter resume."
            if not pdf_reader.pages:
                return False, "PDF contains no readable pages"
        except Exception:
            return False, "Invalid PDF structure"
    
    return True, ""

# Extract text from DOCX files
def extract_text_from_docx(file):
    doc = Document(io.BytesIO(file.read()))
    return "\n".join([para.text for para in doc.paragraphs])

# Enhanced text extraction with section detection and page count warning
@handle_errors
def extract_text_from_pdf(pdf_file) -> Dict:
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        page_count = len(pdf_reader.pages)
        if page_count > 10:
            st.warning(f"Large PDF detected ({page_count} pages). Processing may take longer.")
        
        text = ""
        section_headings = []
        common_sections = [
            'SUMMARY', 'EXPERIENCE', 'EDUCATION', 'SKILLS', 
            'PROJECTS', 'CERTIFICATIONS', 'ACHIEVEMENTS'
        ]
        heading_pattern = re.compile(r'^(' + '|'.join(common_sections) + r')', re.IGNORECASE)
        
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                # Detect section headings
                for line in page_text.split('\n'):
                    if heading_pattern.match(line.strip()):
                        section_headings.append(line.strip().upper())
                text += page_text + "\n"
        
        return {
            "text": text.strip() if text else None,
            "sections": list(set(section_headings)),  # Remove duplicates
            "missing_sections": [s for s in common_sections if s not in [sh.upper() for sh in section_headings]],
            "page_count": page_count
        }
    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}")
        raise

# Extract text from uploaded file with enhanced validation
@handle_errors
def extract_text_from_file(uploaded_file) -> Optional[Dict]:
    try:
        valid, message = validate_file(uploaded_file)
        if not valid:
            st.error(message)
            return None

        if uploaded_file.type == "application/pdf":
            return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(uploaded_file)
            return {
                "text": text,
                "sections": [],
                "missing_sections": [],
                "page_count": 1
            }
        return {
            "text": uploaded_file.read().decode("utf-8").strip(),
            "sections": [],
            "missing_sections": [],
            "page_count": 1
        }
    except Exception as e:
        logger.error(f"File processing failed: {str(e)}")
        raise

# Detect language of text
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# Translate text
def translate_text(text, target_lang="en"):
    if not text or target_lang == "en":
        return text
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        return text

# Generate enhanced analysis prompt with domain-specific considerations
def generate_prompt(resume_text: str, job_role: Optional[str] = None, domain: str = "general", language: str = "en") -> str:
    domain_specific = {
        "tech": "Focus on technical skills, programming languages, and software tools.",
        "healthcare": "Focus on medical certifications, patient care experience, and healthcare technologies.",
        "finance": "Focus on financial modeling, analytical skills, and relevant certifications.",
        "education": "Focus on teaching experience, curriculum development, and educational qualifications.",
        "marketing": "Focus on campaign results, digital marketing skills, and analytics tools."
    }.get(domain, "Consider general professional skills and experiences.")
    
    language_note = f"The resume is in {language.upper()}. Provide feedback in the same language." if language != "en" else ""
    
    base_prompt = f"""
You are a professional resume analyst. Critically evaluate the resume provided below and generate **specific, resume-based** feedback for the role of **{job_role if job_role else "general job applications"}**. Incorporate domain expertise and contextual analysis. Do not provide generic advice—tie every point directly to the resume content. {domain_specific} {language_note}

Structure your response EXACTLY as follows:

---

### ✅ Key Strengths (3–5 bullet points)
- [Identify and quote or paraphrase strong points from the resume]
- [Explain why these points are relevant or impressive for {job_role if job_role else "target roles"}]

### ⚠️ Critical Weaknesses (3–5 bullet points)
- [Cite exact phrases or sections that are weak, vague, or unconvincing]
- [Provide precise, actionable rewrite suggestions]

### 📌 Experience Analysis
- **Relevance to {job_role if job_role else "target roles"}**
  - [Assess how well the experience aligns with the responsibilities or skills expected]
- **Impact Quantification**
  - [Suggest metrics or data points if any bullet lacks measurable outcomes]
- **Chronological Consistency**
  - [Call out any missing dates, gaps, or order issues]

### 📂 Section Analysis
- **Missing Sections**
  - [State clearly what’s missing — e.g., Skills, Projects, Summary]
- **Sections Needing Improvement**
  - [Name specific sections and describe what to improve and how]

### 🤖 ATS Optimization
- **Keyword Alignment with {job_role if job_role else "target roles"}**
  - [Identify missing or underused industry-specific terms]
- **Estimated ATS Score**: [Score out of 100]
- **ATS Formatting Concerns**
  - [Highlight tables, columns, fonts, or symbols that may break parsing]

### 🧠 Content Clarity & Impact
- **Phrases to Improve**
  - [Quote unclear or weak lines and rewrite them]
- **Suggested Power Verbs**
  - [List 5–10 stronger action verbs tailored to their experience]
- **Quantifiable Achievements to Highlight**
  - [Point out any generic bullets and suggest quantified rewrites]

### 🧩 Soft Skills Detected
- [List inferred soft skills like Leadership, Communication, etc., with confidence levels (High/Medium/Low)]

### 🔍 Bias & Inclusion Analysis
- **Potential Biases**
  - [Age, gendered language, or location-based exclusions]
- **Inclusive Language Suggestions**
  - [Improve tone and neutrality if needed]

### ⭐ Overall Rating
**X/10** (e.g., “8.0/10”)  
[Give a detailed explanation justifying this score based on resume quality and readiness for {job_role if job_role else "general applications"}]

### 🎯 Suggested Role: [Most fitting role based on resume content]
- **Alternative Roles**: [If applicable]

### 🛠️ Actionable Recommendations (Priority-Ordered)
1. [Top priority improvement]
2. [Second priority improvement]
3. [Continue if more relevant steps exist]

---

Resume Content:
{resume_text}"""
    return base_prompt

# Improved feedback extraction with robust rating extraction
def extract_summary_info(feedback: str) -> Dict:
    # Enhanced rating extraction with multiple pattern matching
    rating = "N/A"
    rating_patterns = [
        r'Overall Rating:\s*(\d+\.?\d*)\s*/\s*10',  # "Overall Rating: 8.5/10"
        r'Rating:\s*(\d+\.?\d*)\s*out of 10',       # "Rating: 8 out of 10"
        r'Score:\s*(\d+\.?\d*)\s*/10',              # "Score: 7.5/10"
        r'(\d+\.?\d*)\s*\/\s*10\b',                 # "8.5 / 10"
        r'Overall:\s*(\d+\.?\d*)\s*/10',            # "Overall: 7/10"
        r'^(\d+\.?\d*)\s*/10\b'                     # "9/10 at start of line"
    ]
    
    for pattern in rating_patterns:
        rating_match = re.search(pattern, feedback, re.IGNORECASE)
        if rating_match:
            rating = rating_match.group(1)
            break
    
    # Fallback: Search for any number followed by /10 in the rating section
    if rating == "N/A":
        rating_section = re.search(r'(?:Overall Rating|Rating|Score).*?(?=\n###|\n\*\*|\n\n|$)', feedback, re.IGNORECASE | re.DOTALL)
        if rating_section:
            fallback_match = re.search(r'(\d+\.?\d*)\s*/10', rating_section.group(0))
            if fallback_match:
                rating = fallback_match.group(1)
    
    # Final fallback: Calculate from strengths/weaknesses if still not found
    if rating == "N/A":
        strengths = len(re.findall(r'Strength:', feedback, re.IGNORECASE))
        weaknesses = len(re.findall(r'Weakness:', feedback, re.IGNORECASE))
        rating = str(round(5 + (strengths - weaknesses) * 0.5, 1))

    # Extract ATS score with more robust patterns
    ats_score = "0"
    ats_patterns = [
        r'ATS\s*[Ss]core\s*:\s*(\d+)',  # "ATS Score: 85"
        r'ATS\s*compatibility\s*:\s*(\d+)',  # "ATS compatibility: 85"
        r'ATS\s*:\s*(\d+)',  # "ATS: 85"
        r'ATS\s*[Ss]core\s*[iI]s\s*(\d+)',  # "ATS score is 85"
        r'ATS\s*[Oo]ptimization.*?(\d+)\s*\/\s*100',  # "ATS Optimization... 85/100"
        r'Estimated\s*ATS\s*Score\s*:\s*(\d+)'  # "Estimated ATS Score: 85"
    ]

    for pattern in ats_patterns:
        match = re.search(pattern, feedback, re.IGNORECASE)
        if match:
            ats_score = match.group(1)
            break

     # Fallback: Look for any number followed by /100 in the ATS section
    if ats_score == "0":
        ats_section = re.search(r'(?:ATS|Applicant Tracking System).*?(?=\n###|\n\*\*|\n\n|$)', feedback, re.IGNORECASE | re.DOTALL)
        if ats_section:
            fallback_match = re.search(r'(\d+)\s*\/\s*100', ats_section.group(0))
            if fallback_match:
                ats_score = fallback_match.group(1)
    
    # Final fallback: Calculate from keywords if still not found
    if ats_score == "0":
        keyword_matches = len(re.findall(r'keyword', feedback, re.IGNORECASE))
        section_matches = len(re.findall(r'section', feedback, re.IGNORECASE))
        ats_score = str(min(100, keyword_matches * 5 + section_matches * 3))

    # Extract job suggestion
    job_match = re.search(
        r'Suggested Role:\s*(.+?)(?=\n\d+\.|\n[A-Z]|\n\n|$)',
        feedback,
        re.IGNORECASE | re.DOTALL
    )
    job_suggestion = job_match.group(1).strip() if job_match else "Not specified"

    # Extract strengths
    strength_section = ""
    strength_start = re.search(
        r'(?:###\s*Key\s*Strengths|###\s*Strengths?|\*\*Strengths?\*\*:?|Strengths?\s*:\s*)',
        feedback,
        re.IGNORECASE
    )
    if strength_start:
        start_idx = strength_start.end()
        end_idx = re.search(
            r'\n\s*(?:###|\*\*|Critical Weaknesses?)',
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

    # Extract actionable recommendations
    action_match = re.search(
        r'Actionable Recommendations.*?\n(.*?)(?=\n###|\n\*\*|\n\n|$)',
        feedback,
        re.IGNORECASE | re.DOTALL
    )
    actions = []
    if action_match:
        action_lines = action_match.group(1).strip().split('\n')
        actions = [line.strip('1234567890.- ') for line in action_lines if line.strip()]

    # Extract soft skills
    soft_skills_section = ""
    soft_skills_start = re.search(
        r'(?:###\s*Soft\s*Skills|###\s*Skills?|\*\*Soft Skills?\*\*:?|Soft Skills?\s*:\s*)',
        feedback,
        re.IGNORECASE
    )
    if soft_skills_start:
        start_idx = soft_skills_start.end()
        end_idx = re.search(
            r'\n\s*(?:###|\*\*|Bias Analysis)',
            feedback[start_idx:], re.IGNORECASE
        )
        soft_skills_section = feedback[start_idx:] if not end_idx else feedback[start_idx:start_idx + end_idx.start()]

    soft_skills = []
    if soft_skills_section:
        lines = soft_skills_section.strip().split('\n')
        for line in lines:
            cleaned = line.strip('-•* ').strip()
            if cleaned and not cleaned.lower().startswith(('bias', 'rating')):
                soft_skills.append(cleaned)
            if len(soft_skills) >= 5:
                break

    # Extract bias analysis
    bias_section = ""
    bias_start = re.search(
        r'(?:###\s*Bias\s*Analysis|###\s*Bias|\*\*Bias Analysis?\*\*:?|Bias Analysis?\s*:\s*)',
        feedback,
        re.IGNORECASE
    )
    if bias_start:
        start_idx = bias_start.end()
        end_idx = re.search(
            r'\n\s*(?:###|\*\*|Overall Rating)',
            feedback[start_idx:], re.IGNORECASE
        )
        bias_section = feedback[start_idx:] if not end_idx else feedback[start_idx:start_idx + end_idx.start()]

    bias_notes = []
    if bias_section:
        lines = bias_section.strip().split('\n')
        for line in lines:
            cleaned = line.strip('-•* ').strip()
            if cleaned and not cleaned.lower().startswith(('rating', 'overall')):
                bias_notes.append(cleaned)
            if len(bias_notes) >= 3:
                break

    return {
        "rating": rating,
        "ats_score": ats_score,
        "job_suggestion": job_suggestion,
        "strengths": strengths,
        "actions": actions,
        "soft_skills": soft_skills,
        "bias_notes": bias_notes,
        "raw_feedback": feedback,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

# Get real-time job recommendations from Indeed API (mock implementation)
def get_job_recommendations(resume_text: str, location: str = "", domain: str = "general", limit: int = 5) -> List[Dict]:
    """Get real job listings from Adzuna API with fallback to sample data"""
    try:
        # First try Adzuna API
        params = {
            "app_id": os.getenv("ADZUNA_APP_ID"),
            "app_key": os.getenv("ADZUNA_APP_KEY"),
            "what": domain if domain != "general" else "developer",  # Default search term
            "results_per_page": limit,
        }
        
        if location:
            params["where"] = location
            
        response = requests.get(
            "https://api.adzuna.com/v1/api/jobs/us/search/1",  # US jobs, change country code as needed
            params=params,
            timeout=10
        )
        response.raise_for_status()
        
        jobs = []
        for job in response.json().get("results", [])[:limit]:
            match_score = calculate_match_score(
                resume_text, 
                f"{job.get('title', '')} {job.get('description', '')}"
            )
            
            jobs.append({
                "title": job.get("title", "No title"),
                "company": job.get("company", {}).get("display_name", "Unknown"),
                "location": job.get("location", {}).get("display_name", "Remote"),
                "description": (job.get("description", "No description")[:200] + "...") if job.get("description") else "No description",
                "url": job.get("redirect_url", "#"),
                "match_score": round(match_score, 1),
                "salary": format_salary(job)
            })
            
        return sorted(jobs, key=lambda x: x["match_score"], reverse=True)
        
    except Exception as e:
        logger.error(f"Job API error: {str(e)}")
        return get_fallback_jobs(domain, limit)

def format_salary(job: Dict) -> str:
    """Format salary information from Adzuna response"""
    min_sal = job.get("salary_min")
    max_sal = job.get("salary_max")
    currency = job.get("salary_currency", "USD")
    
    if min_sal and max_sal:
        return f"{min_sal:,.0f}-{max_sal:,.0f} {currency}"
    elif min_sal:
        return f"From {min_sal:,.0f} {currency}"
    elif max_sal:
        return f"Up to {max_sal:,.0f} {currency}"
    return "Salary not specified"
    
def calculate_match_score(resume_text: str, job_description: str) -> float:
    """Calculate similarity between resume and job description"""
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume_text, job_description])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100

def get_fallback_jobs(domain: str, limit: int) -> List[Dict]:
    """Fallback sample data when API fails"""
    domain_jobs = {
        "tech": [
            {
                "title": "Software Engineer", 
                "company": "TechCorp",
                "location": "San Francisco, CA",
                "description": "Looking for a skilled software engineer with Python experience...",
                "url": "https://example.com/techjob1",
                "match_score": 85.0,
                "salary": "90,000-120,000 USD"
            },
            {
                "title": "Data Scientist",
                "company": "DataWorks",
                "location": "Remote",
                "description": "Seeking data scientist with machine learning expertise...",
                "url": "https://example.com/techjob2",
                "match_score": 78.5,
                "salary": "100,000-140,000 USD"
            }
        ],
        "healthcare": [
            {
                "title": "Registered Nurse", 
                "company": "City Hospital",
                "location": "New York, NY",
                "description": "RN position requiring BSN and clinical experience...",
                "url": "https://example.com/healthjob1",
                "match_score": 92.0,
                "salary": "70,000-90,000 USD"
            }
        ],
        "general": [
            {
                "title": "Marketing Manager",
                "company": "Brand Solutions",
                "location": "Chicago, IL",
                "description": "Digital marketing role requiring SEO and analytics skills...",
                "url": "https://example.com/genjob1",
                "match_score": 65.0,
                "salary": "60,000-80,000 USD"
            }
        ]
    }
    return domain_jobs.get(domain, domain_jobs["general"])[:limit]

# Resume templates by industry
RESUME_TEMPLATES = {
    "tech": {
        "name": "Technical Professional",
        "sections": ["Summary", "Technical Skills", "Experience", "Projects", "Education", "Certifications"],
        "content": """SUMMARY
Highly skilled Software Engineer with 5+ years of experience in full-stack development...

TECHNICAL SKILLS
- Programming: Python, JavaScript, Java
- Frameworks: React, Node.js, Django
- Tools: Git, Docker, AWS

EXPERIENCE
Senior Software Engineer, TechCompany (2020-Present)
- Developed scalable microservices architecture...
- Led team of 5 developers...

PROJECTS
E-Commerce Platform (2022)
- Built using React and Node.js...
- Improved checkout conversion by 30%...

EDUCATION
B.S. Computer Science, University X (2019)"""
    },
    "healthcare": {
        "name": "Healthcare Professional",
        "sections": ["Professional Summary", "Clinical Experience", "Education", "Licenses & Certifications", "Skills"],
        "content": """PROFESSIONAL SUMMARY
Compassionate Registered Nurse with 7 years of experience in...

CLINICAL EXPERIENCE
Staff Nurse, City Hospital (2018-Present)
- Provided direct patient care to 5-7 patients per shift...
- Administered medications and treatments...

EDUCATION
Bachelor of Science in Nursing, University Y (2017)

LICENSES & CERTIFICATIONS
- Registered Nurse (RN), State License
- BLS, ACLS Certified

SKILLS
- Patient Assessment
- Electronic Health Records
- Emergency Response"""
    },
    "finance": {
        "name": "Finance Professional",
        "sections": ["Profile", "Core Competencies", "Professional Experience", "Education", "Technical Skills"],
        "content": """PROFILE
Results-driven Financial Analyst with expertise in...

CORE COMPETENCIES
- Financial Modeling
- Data Analysis
- Risk Assessment

PROFESSIONAL EXPERIENCE
Financial Analyst, Investment Firm (2019-Present)
- Developed financial models that improved...
- Analyzed portfolio performance...

EDUCATION
MBA in Finance, University Z (2018)

TECHNICAL SKILLS
- Excel (Advanced)
- SQL
- Bloomberg Terminal"""
    }
}

# Async function for API call with better error handling
async def get_ai_feedback_async(client, model_choice, messages, temperature):
    try:
        with ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                executor,
                lambda: client.chat.completions.create(
                    model=model_choice,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=2000,
                    top_p=0.9
                )
            )
            return response
    except Exception as e:
        logger.error(f"API call failed: {str(e)}")
        raise

# UI Components
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
                st.text(st.session_state.extracted_text["text"][:2000] + "...")  # Show first 2000 chars
            else:
                st.warning("Text not extracted yet. Click 'Analyze My Resume' first.")

    return uploaded_file, job_role

def show_section_analysis(extracted_data):
    """Display section analysis results"""
    if st.session_state.get("section_analysis_shown"):
        return  # skip if already shown

    st.session_state["section_analysis_shown"] = True

    with st.expander("Section Analysis", expanded=True):
        cols = st.columns(2)
        if extracted_data["sections"]:
            with cols[0]:
                st.markdown("<h4>Detected Sections</h4>", unsafe_allow_html=True)
                for section in extracted_data["sections"]:
                    st.markdown(f'<div class="section-present">{section}</div>', unsafe_allow_html=True)
        
        if extracted_data["missing_sections"]:
            with cols[1]:
                st.markdown("<h4>Recommended Additions</h4>", unsafe_allow_html=True)
                for section in extracted_data["missing_sections"]:
                    st.markdown(f'<div class="section-missing">{section}</div>', unsafe_allow_html=True)

def show_analysis_results(feedback_data):
    """Display the analysis results in tabs"""
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Summary", "📝 Full Feedback", "🔍 ATS Analysis", "✅ Action Items", "💼 Job Matches", "🔄 History"])
    
    with tab1:
        st.markdown("<h3 style='color: var(--accent);'>Quick Summary</h3>", unsafe_allow_html=True)
    
        # Handle rating display with fallbacks
        rating = feedback_data["rating"]
        if rating == "N/A":
            st.warning("Could not determine exact rating - showing estimated score")
            # Calculate from strengths/weaknesses as fallback
            strengths = len(feedback_data.get("strengths", []))
            weaknesses = len(re.findall(r'Weakness:', feedback_data["raw_feedback"], re.IGNORECASE))
            rating = str(min(10, max(1, round(5 + (strengths - weaknesses) * 0.5, 1))))

        col1, col2 = st.columns([1, 2])
        with col1:
            # Rating card
            st.markdown(f"""
            <div class="feedback-card">
                <h4 style="text-align: center; color: var(--accent);">Overall Rating</h4>
                <div class="rating-circle">{feedback_data["rating"]}/10</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Job suggestion card
            st.markdown(f"""
            <div class="feedback-card">
                <h4 style="color: var(--accent);">Suggested Role</h4>
                <p style="font-size: 16px;">{feedback_data["job_suggestion"]}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Strengths card
            if feedback_data["strengths"]:
                strengths_html = "<ul>" + "".join([f"<li>{s}</li>" for s in feedback_data["strengths"]]) + "</ul>"
                st.markdown(f"""
                <div class="feedback-card">
                    <h4 style="color: var(--accent);">Key Strengths</h4>
                    {strengths_html}
                </div>
                """, unsafe_allow_html=True)
            
            # Weaknesses card
            if "Critical Weaknesses" in feedback_data["raw_feedback"]:
                weakness_start = feedback_data["raw_feedback"].find("Critical Weaknesses")
                weakness_end = feedback_data["raw_feedback"].find("###", weakness_start)
                weakness_section = feedback_data["raw_feedback"][weakness_start:weakness_end].split('\n')[1:]
                weaknesses = [line.strip('-•* ') for line in weakness_section if line.strip() and not line.startswith('###')]
                
                if weaknesses:
                    weaknesses_html = "<ul>" + "".join([f"<li>{w}</li>" for w in weaknesses[:3]]) + "</ul>"
                    st.markdown(f"""
                    <div class="feedback-card">
                        <h4 style="color: var(--accent);">Key Areas for Improvement</h4>
                        {weaknesses_html}
                    </div>
                    """, unsafe_allow_html=True)

    with tab2:
        st.markdown("<h3 style='color: var(--accent);'>Detailed Feedback</h3>", unsafe_allow_html=True)
        st.markdown(feedback_data["raw_feedback"])

    with tab3:
        st.markdown("<h3 style='color: var(--accent);'>ATS Optimization</h3>", unsafe_allow_html=True)
        
        if feedback_data["ats_score"]:
            score = int(feedback_data["ats_score"])
            st.markdown(f"""
            <div class="feedback-card">
                <h4 style="color: var(--accent);">ATS Compatibility Score</h4>
                <div class="ats-meter">
                    <div class="ats-fill" style="width:{score}%"></div>
                </div>
                <p style="text-align: center; font-size: 18px; margin-top: 5px;">{score}/100</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Extract ATS tips from feedback
        ats_start = feedback_data["raw_feedback"].find("ATS Optimization")
        if ats_start != -1:
            ats_end = feedback_data["raw_feedback"].find("###", ats_start + 1)
            ats_section = feedback_data["raw_feedback"][ats_start:ats_end].split('\n')[1:]
            ats_tips = [line.strip('-•* ') for line in ats_section if line.strip() and not line.startswith('###')]
            
            if ats_tips:
                st.markdown("""
                <div class="feedback-card">
                    <h4 style="color: var(--accent);">Top ATS Optimization Tips</h4>
                    <ul>
                """, unsafe_allow_html=True)
                
                for tip in ats_tips[:5]:
                    st.markdown(f"<li>{tip}</li>", unsafe_allow_html=True)
                
                st.markdown("</ul></div>", unsafe_allow_html=True)

    with tab4:
        st.markdown("<h3 style='color: var(--accent);'>Actionable Recommendations</h3>", unsafe_allow_html=True)
        
        if feedback_data["actions"]:
            st.markdown("""
            <div class="feedback-card">
                <h4 style="color: var(--accent);">Priority Improvements</h4>
                <ol>
            """, unsafe_allow_html=True)
            
            for action in feedback_data["actions"][:5]:
                st.markdown(f"<li>{action}</li>", unsafe_allow_html=True)
            
            st.markdown("</ol></div>", unsafe_allow_html=True)
        
        # Add keyword optimization section
        if "keyword" in feedback_data["raw_feedback"].lower():
            keyword_start = feedback_data["raw_feedback"].lower().find("keyword")
            if keyword_start != -1:
                keyword_section = feedback_data["raw_feedback"][keyword_start:keyword_start+500]
                st.markdown("""
                <div class="feedback-card">
                    <h4 style="color: var(--accent);">Keyword Optimization</h4>
                    <p>These keywords are important for your target role:</p>
                """, unsafe_allow_html=True)
                
                # Extract keywords
                keywords = re.findall(r'"(.*?)"', keyword_section)
                if keywords:
                    keyword_display = ", ".join([f'<span class="keyword-match">{k}</span>' for k in keywords[:10]])
                    st.markdown(f"<p>{keyword_display}</p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Soft skills section
        if feedback_data["soft_skills"]:
            st.markdown("""
            <div class="feedback-card">
                <h4 style="color: var(--accent);">Detected Soft Skills</h4>
                <div>
            """, unsafe_allow_html=True)
            
            for skill in feedback_data["soft_skills"][:10]:
                st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Bias analysis section
        if feedback_data["bias_notes"]:
            st.markdown("""
            <div class="feedback-card">
                <h4 style="color: var(--accent);">Bias Analysis</h4>
                <ul>
            """, unsafe_allow_html=True)
            
            for note in feedback_data["bias_notes"][:3]:
                st.markdown(f"<li>{note}</li>", unsafe_allow_html=True)
            
            st.markdown("</ul></div>", unsafe_allow_html=True)

    with tab5:  # Job Matches tab - ONLY place job recommendations appear
        st.markdown("<h3 style='color: var(--accent);'>Real Job Recommendations</h3>", unsafe_allow_html=True)
        
        if not st.session_state.job_matches:
            with st.spinner("Searching for matching jobs..."):
                st.session_state.job_matches = get_job_recommendations(
                    st.session_state.extracted_text["text"],
                    domain=st.session_state.domain
                )
        
        if st.session_state.job_matches:
            for job in st.session_state.job_matches:
                st.markdown(f"""
                <div class="job-card">
                    <h4>{job['title']}</h4>
                    <p><strong>{job['company']}</strong> • {job['location']}</p>
                    <p>{job['description'][:200]}...</p>
                    <p>Salary: {job.get('salary', 'Not specified')}</p>
                    <p class="job-match-score">Match Score: {job['match_score']}%</p>
                    <a href="{job['url']}" target="_blank" class="stButton">View Job</a>
                </div>
                """, unsafe_allow_html=True)
                
            # User feedback on recommendations
            st.markdown("### Were these recommendations helpful?")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("👍 Yes", key="job_feedback_yes"):
                    st.session_state.user_feedback["job_recommendations"] = "positive"
                    st.success("Thanks for your feedback!")
            with col2:
                if st.button("👎 No", key="job_feedback_no"):
                    st.session_state.user_feedback["job_recommendations"] = "negative"
                    st.warning("We'll try to improve future recommendations.")
            with col3:
                if st.button("🤔 Somewhat", key="job_feedback_somewhat"):
                    st.session_state.user_feedback["job_recommendations"] = "neutral"
                    st.info("Thanks for your feedback!")
        else:
            st.warning("No job matches found. Try adjusting your search criteria.")

    with tab6:
        st.markdown("<h3 style='color: var(--accent);'>Resume History</h3>", unsafe_allow_html=True)
        
        if st.session_state.resume_history:
            for i, item in enumerate(st.session_state.resume_history):
                with st.expander(f"Analysis from {item['timestamp']} - Rating: {item['rating']}/10"):
                    st.markdown(f"**Suggested Role:** {item['job_suggestion']}")
                    st.markdown(f"**ATS Score:** {item['ats_score']}/100")
                    
                    # Show comparison if previous analysis exists
                    if i > 0:
                        prev_rating = st.session_state.resume_history[i-1]['rating']
                        rating_diff = float(item['rating']) - float(prev_rating)
                        if rating_diff > 0:
                            st.success(f"Improved by {abs(rating_diff):.1f} points from previous version")
                        elif rating_diff < 0:
                            st.warning(f"Decreased by {abs(rating_diff):.1f} points from previous version")
                        else:
                            st.info("No change from previous version")
                    
                    if st.button(f"View Full Analysis #{i+1}"):
                        st.session_state.feedback_data = item
                        st.session_state.analysis_complete = True
                        st.rerun()
        else:
            st.info("No previous analyses found. Your future analyses will appear here.")

    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            label="📥 Download Feedback",
            data=feedback_data["raw_feedback"],
            file_name="resume_feedback.md",
            mime="text/markdown",
            use_container_width=True
        )
    with col2:
        if st.button("🔄 Analyze Another Resume", use_container_width=True):
            st.session_state.analysis_complete = False
            st.session_state.feedback_data = None
            st.session_state.extracted_text = None
            st.rerun()
    with col3:
        if st.button("✉️ Email Me This Feedback", use_container_width=True):
            st.info("This feature would connect to your email service in a production app")

def show_ats_explanation(score):
    """Display detailed explanation of ATS score"""
    with st.expander("ℹ️ About ATS Scores"):
        st.markdown("""
        **Applicant Tracking System (ATS) Score Explained:**
        
        - **90-100**: Excellent optimization, likely to pass most ATS filters
        - **75-89**: Good optimization, should pass many ATS filters
        - **60-74**: Fair optimization, may need improvements
        - **Below 60**: Poor optimization, unlikely to pass ATS filters
        
        **Key Factors Affecting ATS Score:**
        - Keyword matching with job description
        - Proper section headers
        - Standard formatting
        - Avoidance of complex layouts/tables
        - Appropriate length
        """)
        
        if score is not None:
            if score >= 90:
                st.success("This resume is well-optimized for ATS systems!")
            elif score >= 75:
                st.info("This resume is reasonably well-optimized but could be improved")
            elif score >= 60:
                st.warning("This resume needs optimization for better ATS performance")
            else:
                st.error("This resume needs significant optimization for ATS systems")

def calculate_resume_similarity(resume1, resume2):
    """Calculate similarity between two resumes"""
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume1, resume2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100

def show_similarity_analysis(selected_data):
    """Display similarity matrix for selected resumes"""
    st.markdown("### Resume Similarity Analysis")
    
    # Create similarity matrix
    names = [r['name'] for r in selected_data]
    similarity_matrix = np.zeros((len(selected_data), len(selected_data)))
    
    for i in range(len(selected_data)):
        for j in range(len(selected_data)):
            if i == j:
                similarity_matrix[i][j] = 100
            elif i < j:
                similarity = calculate_resume_similarity(
                    selected_data[i]['extracted_data']['text'],
                    selected_data[j]['extracted_data']['text']
                )
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
    
    # Display as heatmap
    fig = px.imshow(
        similarity_matrix,
        labels=dict(x="Resume", y="Resume", color="Similarity"),
        x=names,
        y=names,
        text_auto=".1f",
        color_continuous_scale="Viridis",
        title="Resume Similarity Matrix (%)"
    )
    st.plotly_chart(fig, use_container_width=True)

def generate_quality_scorecard(resume):
    """Generate a visual quality scorecard for a resume"""
    if not resume['analysis']:
        return None
    
    scores = {
        "Content": float(resume['analysis']['rating']) * 10,
        "ATS": float(resume['analysis']['ats_score']) if resume['analysis']['ats_score'] else 0,
        "Structure": min(100, len(resume['metadata']['sections']) * 15),
        "Clarity": min(100, resume['metadata']['word_count'] / 10)
    }
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=list(scores.values()),
        theta=list(scores.keys()),
        fill='toself',
        name='Quality Score'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="Resume Quality Scorecard"
    )
    
    return fig

def show_job_description_matching(selected_data):
    """Show how well resumes match a specific job description"""
    st.markdown("### Job Description Matching")
    
    job_desc = st.text_area(
        "Paste a job description to compare against",
        height=200,
        key="job_desc_comparison"
    )
    
    if job_desc and len(job_desc) > 50:
        matches = []
        for resume in selected_data:
            match_score = calculate_match_score(
                resume['extracted_data']['text'],
                job_desc
            )
            matches.append({
                "Name": resume['name'],
                "Match Score": match_score,
                "Rating": float(resume['analysis']['rating']) if resume['analysis'] and resume['analysis']['rating'] != "N/A" else 0,
                "ATS": float(resume['analysis']['ats_score']) if resume['analysis'] and resume['analysis']['ats_score'] else 0
            })
        
        match_df = pd.DataFrame(matches).set_index("Name")
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(
                match_df,
                use_container_width=True,
                column_config={
                    "Match Score": st.column_config.ProgressColumn(
                        "Match %",
                        format="%.1f",
                        min_value=0,
                        max_value=100
                    )
                }
            )
        
        with col2:
            fig = px.bar(
                match_df.reset_index(),
                x="Name",
                y="Match Score",
                title="Job Description Match Scores",
                color="Match Score",
                text="Match Score"
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

def show_resume_templates():
    """Display resume templates for different industries"""
    st.markdown("<h3 style='color: var(--accent);'>Resume Templates</h3>", unsafe_allow_html=True)
    
    selected_domain = st.selectbox(
        "Select Industry",
        ["tech", "healthcare", "finance"],
        format_func=lambda x: x.capitalize(),
        key="template_domain"
    )
    
    if selected_domain in RESUME_TEMPLATES:
        template = RESUME_TEMPLATES[selected_domain]
        st.session_state.selected_template = template
        
        st.markdown(f"<h4>{template['name']} Template</h4>", unsafe_allow_html=True)
        st.markdown("<div class='template-preview'>" + template["content"].replace("\n", "<br>") + "</div>", unsafe_allow_html=True)
        
        if st.button("Use This Template"):
            st.session_state.extracted_text = {
                "text": template["content"],
                "sections": template["sections"],
                "missing_sections": [],
                "page_count": 1
            }
            st.success("Template loaded! You can now analyze it or edit the text.")

def show_language_settings():
    """Display language selection options"""
    st.markdown("<h3 style='color: var(--accent);'>Language Settings</h3>", unsafe_allow_html=True)
    
    language = st.selectbox(
        "Select Language",
        ["en", "es", "fr", "de", "pt", "it", "nl", "ru", "zh", "ja"],
        format_func=lambda x: {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "pt": "Portuguese",
            "it": "Italian",
            "nl": "Dutch",
            "ru": "Russian",
            "zh": "Chinese",
            "ja": "Japanese"
        }[x],
        key="language_selector"
    )
    
    st.session_state.language = language
    st.info(f"Analysis will be provided in {language.upper()} if supported")

def show_domain_settings():
    """Display domain/industry selection options"""
    st.markdown("<h3 style='color: var(--accent);'>Industry Focus</h3>", unsafe_allow_html=True)
    
    domain = st.selectbox(
        "Select Industry",
        ["general", "tech", "healthcare", "finance", "education", "marketing"],
        format_func=lambda x: x.capitalize(),
        key="domain_selector"
    )
    
    st.session_state.domain = domain
    st.info(f"Analysis will be tailored for {domain.capitalize()} industry")

# Main app function
def main():
    # Header section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("AI Resume Critiquer Pro")
        st.markdown("""
        <p style="font-size: 18px;">
            Get expert-level feedback on your resume instantly with advanced features including:
            real-time job matching, bias detection, multilingual support, and industry-specific analysis.
        </p>
        """, unsafe_allow_html=True)
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/3132/3132693.png", width=150)

    # Sidebar with enhanced settings
    with st.sidebar:
        st.markdown("<h2 style='color: var(--accent);'>Settings</h2>", unsafe_allow_html=True)
        
        show_language_settings()
        show_domain_settings()
        
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
        
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ["Standard", "Detailed", "Comprehensive"],
            index=1,
            help="Choose how thorough the analysis should be"
        )
        
        st.markdown("---")
        st.markdown("<h4 style='color: var(--accent);'>How it works:</h4>", unsafe_allow_html=True)
        st.markdown("1. Upload your resume (PDF, TXT, DOCX)")
        st.markdown("2. Enter target job (optional)")
        st.markdown("3. Get instant expert feedback")
        st.markdown("4. Explore job matches and templates")
        st.markdown("---")
        st.markdown("<small>Your data is processed securely and not stored permanently.</small>", unsafe_allow_html=True)

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Analyze Resume", "📂 Resume Manager", "👥 Compare Candidates", "💡 Templates"])    
    
    with tab1:
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

                show_section_analysis(extracted_data)

                # Detect language if not English
                if st.session_state.language == "auto":
                    detected_lang = detect_language(extracted_data["text"])
                    st.session_state.language = detected_lang
                    st.info(f"Detected language: {detected_lang.upper()}")
                
                st.session_state.extracted_text = extracted_data
                word_count = len(extracted_data["text"].split())
                
                if word_count < 50:
                    st.warning(f"The resume seems very short ({word_count} words). Please check if text was properly extracted.")
                elif word_count > 1000:
                    st.info(f"Processing large resume ({word_count} words). This may take a moment...")

            # Show section analysis
            show_section_analysis(extracted_data)

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
                            "content": "You are an expert resume reviewer with 10+ years in HR and recruiting. Provide detailed, actionable feedback."
                        },
                        {"role": "user", "content": prompt}
                    ]
                    
                    # Adjust tokens based on analysis depth
                    max_tokens = {
                        "Standard": 1500,
                        "Detailed": 2000,
                        "Comprehensive": 3000
                    }.get(analysis_depth, 2000)
                    
                    response = client.chat.completions.create(
                        model=model_choice,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=0.9
                    )
                    
                    feedback = response.choices[0].message.content
                    
                    # Translate feedback if needed
                    if st.session_state.language != "en":
                        feedback = translate_text(feedback, st.session_state.language)
                    
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
    
    with tab2:
        # New resume management functionality
        handle_multi_file_upload()
        st.markdown("---")
        show_resume_dashboard()
    
    with tab3:
        # New candidate comparison functionality
        show_candidate_comparison()
    
    with tab4:
        # Existing templates functionality
        show_resume_templates()

if __name__ == "__main__":
    main()