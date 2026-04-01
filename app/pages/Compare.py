import streamlit as st
import plotly.express as px
import pandas as pd
import re

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app import utils

# Import get_groq_client if it's defined elsewhere in your project
try:
    from app.utils import get_groq_client
except ImportError:
    def get_groq_client():
        raise NotImplementedError("get_groq_client is not implemented. Please provide its implementation.")

def show_candidate_comparison():
    """Display comparison view for multiple resumes with comprehensive error handling"""
    st.title("👥 Candidate Comparison")
    
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
            print(f"Skipping resume {resume.get('name', 'unnamed')} due to: {str(e)}")
            print(f"Problematic resume data: {resume}")
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
                            # Define or import generate_prompt as needed
                            def generate_prompt(text, domain="", language="en"):
                                return f"Please review the following resume text for the domain '{domain}' in language '{language}':\n\n{text}"

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

                            # Define extract_summary_info to parse feedback into a summary dictionary
                            def extract_summary_info(feedback_text):
                                # Dummy implementation: replace with actual parsing logic as needed
                                return {
                                    "rating": "N/A",
                                    "ats_score": "N/A",
                                    "job_suggestion": "N/A",
                                    "raw_feedback": feedback_text
                                }

                            from datetime import datetime
                            summary_info = extract_summary_info(feedback)
                            summary_info['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M")
                            resume['analysis'] = summary_info
                            st.success("Re-analysis complete!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to re-analyze: {str(e)}")

def init_session_state():
    """Initialize Streamlit session state variables if not already set."""
    if 'uploaded_resumes' not in st.session_state:
        st.session_state.uploaded_resumes = []
    if 'resume_tags' not in st.session_state:
        st.session_state.resume_tags = {}
    if 'domain' not in st.session_state:
        st.session_state.domain = ""
    if 'language' not in st.session_state:
        st.session_state.language = "en"

def main():
    init_session_state()
    show_candidate_comparison()

if __name__ == "__main__":
    main()