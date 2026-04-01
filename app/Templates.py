import streamlit as st
import sys
import os
from fpdf import FPDF
from io import BytesIO
from utils import init_session_state, load_css

# Add the root of the project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Resume templates by industry with enhanced structure
RESUME_TEMPLATES = {
    "tech": {
        "name": "Technical Professional",
        "icon": "💻",
        "description": "Optimized for software engineers, developers, and IT professionals",
        "sections": ["Summary", "Technical Skills", "Experience", "Projects", "Education", "Certifications"],
        "content": {
            "Summary": "Highly skilled Software Engineer with 5+ years of experience in full-stack development. Specialized in building scalable web applications and microservices architectures.",
            "Technical Skills": [
                "Programming: Python, JavaScript, Java, TypeScript",
                "Frameworks: React, Node.js, Django, Spring Boot",
                "Tools: Git, Docker, Kubernetes, AWS, CI/CD pipelines"
            ],
            "Experience": [
                "Senior Software Engineer, TechCompany (2020-Present)",
                "- Designed and implemented scalable microservices architecture serving 1M+ users",
                "- Led team of 5 developers in agile environment, improving deployment frequency by 40%",
                "- Optimized database queries reducing API response times by 65%"
            ],
            "Projects": [
                "E-Commerce Platform (2022)",
                "- Built using React and Node.js with Redux for state management",
                "- Implemented responsive design improving mobile conversion by 30%",
                "- Integrated payment gateways and inventory management systems"
            ],
            "Education": "B.S. Computer Science, University X (2019)\nGPA: 3.8/4.0",
            "Certifications": [
                "AWS Certified Solutions Architect - Associate (2022)",
                "Google Professional Data Engineer (2021)"
            ]
        }
    },
    "healthcare": {
        "name": "Healthcare Professional",
        "icon": "🏥",
        "description": "Designed for nurses, doctors, and medical practitioners",
        "sections": ["Professional Summary", "Clinical Experience", "Education", "Licenses & Certifications", "Skills"],
        "content": {
            "Professional Summary": "Compassionate Registered Nurse with 7 years of experience in acute care settings. Proven ability to deliver high-quality patient care and collaborate with multidisciplinary teams.",
            "Clinical Experience": [
                "Staff Nurse, City Hospital (2018-Present)",
                "- Provided direct patient care to 5-7 patients per shift in 40-bed medical-surgical unit",
                "- Administered medications and treatments with 100% accuracy rate",
                "- Trained 15+ new nurses on hospital protocols and EHR systems"
            ],
            "Education": "Bachelor of Science in Nursing, University Y (2017)\nMinor in Public Health",
            "Licenses & Certifications": [
                "Registered Nurse (RN), State License #123456 (Active)",
                "Basic Life Support (BLS) Certification (Expires 2024)",
                "Advanced Cardiac Life Support (ACLS) Certification"
            ],
            "Skills": [
                "Patient Assessment and Monitoring",
                "Electronic Health Records (Epic, Cerner)",
                "Emergency Response and Crisis Management",
                "Patient and Family Education"
            ]
        }
    },
    "finance": {
        "name": "Finance Professional",
        "icon": "💰",
        "description": "Tailored for financial analysts, accountants, and investment professionals",
        "sections": ["Profile", "Core Competencies", "Professional Experience", "Education", "Technical Skills"],
        "content": {
            "Profile": "Results-driven Financial Analyst with expertise in financial modeling, data analysis, and investment strategies. Adept at translating complex financial data into actionable business insights.",
            "Core Competencies": [
                "Financial Modeling & Valuation",
                "Data Analysis & Visualization",
                "Risk Assessment & Management",
                "Portfolio Optimization",
                "Regulatory Compliance"
            ],
            "Professional Experience": [
                "Financial Analyst, Investment Firm (2019-Present)",
                "- Developed financial models that improved investment decision accuracy by 25%",
                "- Analyzed portfolio performance and recommended reallocation strategies",
                "- Prepared quarterly reports for executive leadership and stakeholders"
            ],
            "Education": "MBA in Finance, University Z (2018)\nB.S. in Economics, University A (2016)",
            "Technical Skills": [
                "Advanced Excel (VBA, Power Query)",
                "SQL & Database Management",
                "Bloomberg Terminal & FactSet",
                "Tableau & Power BI",
                "Python for Financial Analysis"
            ]
        }
    }
}

def generate_pdf(text: str, title: str = "Resume") -> BytesIO:
    """Generate a PDF from text content"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    
    # Add title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt=title, ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    
    # Process sections
    lines = text.split('\n')
    for line in lines:
        if line.strip().endswith(':'):  # Section header detection
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, txt=line, ln=True)
            pdf.set_font("Arial", size=12)
        else:
            pdf.multi_cell(0, 10, txt=line)
        pdf.ln(2)
    
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

def format_resume_content(template_content: dict) -> str:
    """Convert structured content to plain text with formatting"""
    formatted_text = ""
    for section, content in template_content.items():
        formatted_text += f"{section.upper()}:\n"
        if isinstance(content, list):
            for item in content:
                formatted_text += f"{item}\n"
        else:
            formatted_text += f"{content}\n"
        formatted_text += "\n"
    return formatted_text.strip()

def show_template_selection():
    """Display template selection interface"""
    st.title("📝 Professional Resume Templates")
    st.markdown("""
    <style>
    .template-card {
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background: var(--card);
        box-shadow: var(--shadow-sm);
        transition: all 0.3s ease;
    }
    .template-card:hover {
        box-shadow: var(--shadow);
        transform: translateY(-2px);
    }
    .template-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.5rem;
    }
    .template-icon {
        font-size: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    cols = st.columns(3)
    for idx, (domain, template) in enumerate(RESUME_TEMPLATES.items()):
        with cols[idx % 3]:
            st.markdown(f"""
            <div class="template-card" onclick="document.getElementById('select_{domain}').click()">
                <div class="template-header">
                    <div class="template-icon">{template['icon']}</div>
                    <h3>{template['name']}</h3>
                </div>
                <p>{template['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select", key=f"select_{domain}", use_container_width=True):
                st.session_state.selected_domain = domain
                st.rerun()

def show_template_editor():
    """Display the selected template for editing"""
    domain = st.session_state.selected_domain
    template = RESUME_TEMPLATES[domain]
    
    st.title(f"{template['icon']} {template['name']} Template")
    st.caption(template['description'])
    
    with st.expander("ℹ️ Template Guidelines", expanded=True):
        st.markdown(f"""
        - **Best for**: {domain.capitalize()} industry professionals
        - **Recommended sections**: {", ".join(template['sections'])}
        - **Tips**: Keep bullet points concise, use action verbs, quantify achievements
        """)
    
    # Convert structured content to editable text
    default_content = format_resume_content(template['content'])
    edited_resume = st.text_area(
        "Edit your resume below:", 
        value=default_content, 
        height=500,
        key=f"editor_{domain}"
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✨ Use This Template", use_container_width=True):
            st.session_state.extracted_text = {
                "text": edited_resume,
                "sections": template['sections'],
                "missing_sections": [],
                "page_count": 1
            }
            st.success("Template loaded successfully!")
            st.switch_page("pages/1_📄_Analyze.py")
    
    with col2:
        st.download_button(
            label="📄 Download as Text",
            data=edited_resume,
            file_name=f"{domain}_resume.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col3:
        pdf_file = generate_pdf(edited_resume, f"{template['name']} Resume")
        st.download_button(
            label="📥 Download as PDF",
            data=pdf_file,
            file_name=f"{domain}_resume.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    
    if st.button("← Back to Templates", type="secondary"):
        del st.session_state.selected_domain
        st.rerun()

def show_resume_templates():
    """Main template display function"""
    load_css()  # Load your custom CSS
    
    if 'selected_domain' not in st.session_state:
        show_template_selection()
    else:
        show_template_editor()

def main():
    init_session_state()
    show_resume_templates()

if __name__ == "__main__":
    main()