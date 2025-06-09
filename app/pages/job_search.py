import streamlit as st
import requests
import os
from dotenv import load_dotenv
from dotenv import dotenv_values
from datetime import datetime

# Load environment variables
config = dotenv_values(".env")
os.environ["RAPIDAPI_KEY"] = config.get("RAPIDAPI_KEY", "")

# Streamlit page config — must be first
st.set_page_config(page_title="Job Search", layout="wide")

# Load external CSS styling
try:
    with open("job_styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("⚠️ job_styles.css not found. Custom styling won't be applied.")


def format_date(date_str):
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
        return dt.strftime("%b %d, %Y")
    except:
        return "N/A"


def fetch_jobs(job_title, location=None):
    """Fetch job listings from the active-jobs-db API via RapidAPI"""
    url = "https://active-jobs-db.p.rapidapi.com/active-ats-7d"

    querystring = {
        "limit": "10",
        "offset": "0",
        "title_filter": job_title,
        "location_filter": location or "",
        "description_type": "text"
    }

    headers = {
        "x-rapidapi-key": os.getenv("RAPIDAPI_KEY"),
        "x-rapidapi-host": "active-jobs-db.p.rapidapi.com"
    }

    st.write("Current API Key:", os.getenv("RAPIDAPI_KEY"))


    try:
        response = requests.get(url, headers=headers, params=querystring)
        if response.status_code == 200:
            data = response.json()

            # 🔍 Print data structure once to debug
            

            # ✅ Safe handling if it's a list or dict
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return data.get("data", [])
            else:
                return []

        else:
            st.error(f"API error: {response.status_code} - {response.text}")
            return []

    except Exception as e:
        st.error(f"Request failed: {e}")
        return []


def main():
    st.title("🔍 Job Search")
    st.markdown("Find relevant job openings based on your skills and preferences.")

    with st.form("job_search_form"):
        col1, col2 = st.columns([2, 1])
        with col1:
            job_title = st.text_input("Job Title", placeholder="e.g. Data Engineer")
        with col2:
            location = st.text_input("Location (optional)", placeholder="e.g. United States")

        submitted = st.form_submit_button("Search Jobs")

    if submitted and job_title:
        with st.spinner("🔎 Searching for jobs..."):
            jobs = fetch_jobs(job_title, location)

        if jobs:
            st.success(f"✅ Found {len(jobs)} job openings")
            
            for job in jobs:
                title = job.get("title", "N/A")
                company = job.get("organization", "N/A")
                location = (job.get("locations_derived") or ["Remote/Unknown"])[0]
                posted = format_date(job.get("date_posted", ""))
                apply_link = job.get("url", "#")
                description = job.get("description_text", "")

                with st.container():
                    st.markdown(f"""
                    <div class="job-card">
                        <h4>{title}</h4>
                        <p class="company">🏢 {company}</p>
                        <p class="location">📍 {location}</p>
                        <p>📅 Posted: {posted}</p>
                        <p style="font-size: 0.9rem;">{description[:400]}...</p>
                        <a href="{apply_link}" target="_blank" class="apply-button">Apply Now</a>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No jobs found. Try different search terms.")

    if st.button("← Back to Resume Analyzer"):
        st.switch_page("main.py")


if __name__ == "__main__":
    main()
