import streamlit as st
import pandas as pd
import tempfile
import os
import json
import fitz
from io import BytesIO
from pdf2image import convert_from_path
import pytesseract
from duckduckgo_search import DDGS
from groq import Groq

st.set_page_config(page_title="Paper Lens AI", layout="wide")

st.markdown("""
<style>
    /* Main Background */
    .main {background: #0e1117;}

    /* Header Box Styling */
    .header-box {
        background: linear-gradient(145deg, #1e2130, #161924);
        border: 1px solid #2d3342;
        border-radius: 16px;
        padding: 40px 20px;
        text-align: center;
        margin-bottom: 40px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }

    .header-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 48px;
        font-weight: 300;
        color: #ffffff;
        margin: 0;
        letter-spacing: 1.5px;
    }

    .header-highlight {
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #a29bfe, #6c5ce7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Custom Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #6c5ce7, #8e7bf5);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(108, 92, 231, 0.3);
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(108, 92, 231, 0.5);
    }

    /* Result Card Styling */
    .card {
        background: #1c2029;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2d3342;
        margin-bottom: 10px;
    }

    .verified {color: #00b894; font-weight: bold; background: rgba(0, 184, 148, 0.1); padding: 2px 8px; border-radius: 4px;}
    .unverified {color: #fab1a0; font-weight: bold; background: rgba(250, 177, 160, 0.1); padding: 2px 8px; border-radius: 4px;}

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <h1 class="header-title">Paper <span class="header-highlight">Lens</span> AI</h1>
</div>
""", unsafe_allow_html=True)


def query_groq(prompt: str, api_key: str) -> str:
    if not api_key:
        return ""

    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(

            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stream=False,
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"AI Error: {e}")
        return ""


def clean_json_output(response: str):
    try:
        start = response.find("[")
        end = response.rfind("]") + 1
        if start != -1 and end > start:
            return json.loads(response[start:end])
    except:
        pass
    return None


def extract_text_pymupdf(pdf_path: str) -> str:
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc[:3]:
            text += page.get_text("text") + "\n\n"
        if len(text) > 500:
            return text
    except:
        pass
    return ""


def ocr_fallback(pdf_path: str):
    try:
        images = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=3)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img) + "\n\n"
        return text
    except:
        return ""


def verify_data_web(author_name, affiliation):
    if not affiliation or affiliation == "Not found":
        return "No Affiliation", None

    query = f'"{author_name}" "{affiliation}"'
    try:
        results = DDGS().text(query, max_results=2)
        if results:
            return "Verified", results[0]['href']
    except:
        pass

    return "Unverified", None


def process_paper(text: str, api_key: str):
    prompt = f"""
    You are a research assistant. Extract data from the following academic paper text.

    Task:
    1. Identify the Paper Title.
    2. Identify ALL authors in the correct order.
    3. For each author, extract their specific affiliation (University/Department) and Email.
    4. If an email is listed in a footer, map it to the correct name.

    Return ONLY a JSON array of objects. Do not write any introduction.
    Format:
    [
        {{
            "title": "Paper Title",
            "author_name": "Name",
            "affiliation": "Affiliation or null",
            "email": "Email or null"
        }},
        ...
    ]

    Text Data:
    {text[:16000]}
    """

    response = query_groq(prompt, api_key)
    return clean_json_output(response) or []



with st.sidebar:
    st.markdown("### Settings")

    enable_search = st.toggle("Enable Web Verification", value=True)
    st.caption("Cross-references extracted authors with online sources.")


uploaded_files = st.file_uploader("Upload Files (PDF)", type="pdf", accept_multiple_files=True)

if uploaded_files and st.button("Analyze", use_container_width=True):

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except:
            pass

    if not api_key:
        st.error("Server Configuration Error: GROQ_API_KEY not found in Environment or Secrets.")
        st.stop()

    all_papers_data = []
    total_files = len(uploaded_files)

    file_progress = st.progress(0)
    status_text = st.empty()

    for i, file_obj in enumerate(uploaded_files):
        status_text.markdown(f"**Processing {i + 1}/{total_files}:** `{file_obj.name}`")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(file_obj.getvalue())
            pdf_path = f.name

        text = extract_text_pymupdf(pdf_path)
        if not text:
            text = ocr_fallback(pdf_path)

        raw_data = process_paper(text, api_key)

        if raw_data:
            extracted_title = "Unknown Title"
            for item in raw_data:
                if item.get("title") and str(item.get("title")).strip():
                    extracted_title = item.get("title")
                    break

            num_authors = len(raw_data)
            for idx, entry in enumerate(raw_data):
                role = "Co-Author"
                if idx == 0:
                    role = "First Author"
                elif idx == num_authors - 1 and num_authors > 1:
                    role = "Last Author"

                ver_status, ver_link = "Skipped", None
                if enable_search:
                    ver_status, ver_link = verify_data_web(entry.get('author_name'), entry.get('affiliation'))

                all_papers_data.append({
                    "Source File": file_obj.name,
                    "Paper Title": extracted_title,
                    "Role": role,
                    "Author Name": entry.get('author_name'),
                    "Affiliation": entry.get('affiliation', 'Not found'),
                    "Email": entry.get('email', 'Not found'),
                    "Verification": ver_status,
                    "Source Link": ver_link
                })

        os.unlink(pdf_path)
        file_progress.progress((i + 1) / total_files)

    status_text.success("Batch Processing Complete!")

    if all_papers_data:
        df = pd.DataFrame(all_papers_data)

        with st.expander("View Consolidated Data Table", expanded=True):
            st.dataframe(df, use_container_width=True)

        st.markdown("### Preview (Highlights)")
        for _, row in df.head(5).iterrows():
            if row['Role'] in ["First Author", "Last Author"]:
                border_color = "#6c5ce7" if row['Role'] == "First Author" else "#d63031"
                icon = "🥇" if row['Role'] == "First Author" else "🎓"
                ver_class = "verified" if "Verified" in row['Verification'] else "unverified"

                st.markdown(f"""
                <div class="card" style="border-left: 4px solid {border_color};">
                    <h5 style="margin:0; color:white;">{icon} {row['Role']}: {row['Author Name']} 
                        <span class="{ver_class}" style="font-size:10px; float:right; margin-top:2px;">{row['Verification']}</span>
                    </h5>
                    <p style="color:#b2bec3; margin:5px 0 0 0; font-size:13px;">{row['Affiliation']}</p>
                    <p style="color:#7f8c8d; margin:2px 0 0 0; font-size:12px;"><i>{row['Paper Title'][:50]}...</i></p>
                </div>
                """, unsafe_allow_html=True)

        if len(df) > 5:
            st.caption(f"...and {len(df) - 5} more rows.")

        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Download Merged Excel", buffer.getvalue(), "paper_lens_batch.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with c2:
            st.download_button("Download Merged CSV", df.to_csv(index=False).encode(), "paper_lens_batch.csv",
                               "text/csv")
    else:
        st.warning("No data extracted. Please check if the uploaded files are correct.")