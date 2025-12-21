import io
import pdfplumber
import os
import google.generativeai as genai
import streamlit as st

# Move the API configuration inside the function to avoid import issues
model = None

def get_model():
    """Initialize the model with API key from secrets"""
    global model
    if model is None:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model = genai.GenerativeModel("gemini-2.0-flash")
    return model

def read_report(file) -> str:
    """Return plain text from an uploaded PDF or text file."""
    name = file.name.lower()
    if name.endswith(".pdf"):
        text_chunks = []
        with pdfplumber.open(io.BytesIO(file.read())) as pdf:
            for page in pdf.pages:
                text_chunks.append(page.extract_text() or "")
        return "\n\n".join(text_chunks)
    else:
        # Assume text-like files
        return file.read().decode("utf-8", errors="ignore")

def build_prompt(user_note: str, ticker: str, report_text: str) -> str:
    """Build the complete prompt including the report text."""
    return f"""
As an equity research analyst, analyze the provided financial report and deliver a concise, professional analysis in Markdown format. Present the analysis as a direct, objective report without referencing the process of reviewing excerpts or using a model. Focus on actionable insights, explicitly noting any missing information without speculation. Structure the response as follows:

# Executive Summary
- Provide 3–6 bullets summarizing key findings, focusing on financial performance, strategic developments, and market positioning.

# Key Highlights
- Highlight primary drivers of performance, potential catalysts for growth, and notable operational or strategic achievements.

# Concerns and Risks
- Identify red flags, including issues in accounting practices, liquidity constraints, guidance reliability, or customer/supplier concentration risks.

# Quality of Earnings and Cash Flow
- Analyze working capital trends, free cash flow conversion, and sustainability of earnings.

Context:
- Ticker: {ticker or 'N/A'}
- User Note: {user_note or 'N/A'}

FINANCIAL REPORT:
{report_text}
"""

def analyze_report(file, ticker: str = None, user_note: str = None) -> str:
    """Analyze the uploaded report and return the generated review."""
    try:
        # Get the model instance
        model_instance = get_model()
        
        # Extract text from the file
        raw_text = read_report(file)
        if not raw_text.strip():
            return "Couldn't extract any text from the file."

        # Build the complete prompt with the report text
        prompt = build_prompt(user_note or "", ticker or "", raw_text)
        
        # Generate content with the complete prompt
        resp = model_instance.generate_content(prompt)
        return resp.text or "_No response text returned._"

    except Exception as e:
        return f"Error analyzing the report: {e}"