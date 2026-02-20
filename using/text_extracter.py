#import cloudscraper
import requests
from bs4 import BeautifulSoup
from supabase import create_client, Client
import streamlit as st
import os
import fitz  # PyMuPDF
import re
#from sentence_model import process_single_document, results_to_dataframe

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-AU,en;q=0.9",
}

# Insert your Supabase URL and Key here
url = os.environ.get("SUPABASE_URL", "https://lshtgdpdskhqqxdcwpjo.supabase.co")
key = os.environ.get("SUPABASE_KEY","")

supabase = create_client(url, key)

def get_reports(ticker, year=None, source=None, ASX_200: int = True):
    if ASX_200:
        table_name = "Equity Reports ASX200"
    else:
        table_name = "Equity Reports ASX Small Cap"
    query = supabase.table(table_name).select("*").eq("ticker", ticker)
    if year:
        if isinstance(year, list):
            query = query.in_("year", year)
        else:
            query = query.eq("year", year)

    if source:
        if isinstance(source, list):
            query = query.in_("source", source)
        else:
            query = query.eq("source", source)

    return query.execute().data

def get_GPFS_reports(ticker, year=None):
    table_name= "GPFS"
    query = supabase.table(table_name).select("*").eq("ticker", ticker)
    if year:
        if isinstance(year, list):
            query = query.in_("year", year)
        else:
            query = query.eq("year", year)
    
    return query.execute().data



def get_article_text(url, source: str):

    if source in ["bell_potter","Buy_hold_sell","motely_fool", "live_wire", "ord_minnet","wilsonsadvisory", "morningstar"]:
        session = requests.Session()
        session.headers.update(HEADERS)
        r = session.get(url)
        
        r.raise_for_status()
    """elif source in ["money_of_mine"]:
        scraper = cloudscraper.create_scraper()
        r = scraper.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
    """
    soup = BeautifulSoup(r.text, "html.parser")

    # Remove the footer
    footer = soup.find("footer")
    if footer:
        footer.decompose()  # removes the footer from the soup
    
    # Extract remaining <p> text
    paragraphs = soup.find_all("p")
    text = " ".join(p.get_text() for p in paragraphs)
    
    return text

def extract_text_from_GPFS(url: str) -> str:
    """
    Extracts text content from a GPFS URL which is in pdf form
    """
    try:

        # Download PDF
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        # Open PDF and extract text while avoiding tables
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            full_text = []
            
            for page in doc:
                
                # Find table bounding boxes
                tables = page.find_tables()
                table_bboxes = [tabs.bbox for tabs in tables]
                
                #  Extract text blocks (includes coordinates)
                # Block format: (x0, y0, x1, y1, "text", block_no, block_type)
                blocks = page.get_text("blocks")
                
                for b in blocks:
                    text_rect = fitz.Rect(b[:4])
                    # Only keep text if it doesn't overlap with any table
                    if not any(text_rect.intersects(t_bbox) for t_bbox in table_bboxes):
                        # block_type 0 is text; 1 is image (charts/graphics)
                        if b[6] == 0: 
                            full_text.append(b[4])

            return "\n".join(full_text)

    except Exception as e:
        print(f"Error extracting text from GPFS URL: {e}")
        return ""

    #return text

def clean_GPFS_text(text: str) -> str:
    """
    Cleans the extracted text from GPFS by removing headers, footers, and 
    other non-content elements.
    """
    # Clean text so sentiment is not thrown off by page numbers, headers, footers, and other common PDF artifacts
    cleaned_text = re.sub(r"Page \d+ of \d+", "", text)  # Remove page numbers
    cleaned_text = re.sub(r"^.*?Report Title.*?$", "", cleaned_text, flags=re.MULTILINE)  # Remove report title lines
    cleaned_text = re.sub(r"^.*?Company Name.*?$", "", cleaned_text, flags=re.MULTILINE)  # Remove company name lines

    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip() # Normalize whitespace
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', cleaned_text) # Remove non-ASCII characters, which can often be artifacts from PDF extraction

    
    # Split into lines for numeric/special char filtering
    lines = cleaned_text.split("\n")
    filtered_lines = []
    
    digit_ratio_threshold = 0.3  # Adjust as needed
    special_char_threshold = 0.3  # Adjust as needed

    for line in lines:
        line = line.strip()

        # Remove lines that are mostly numbers (numeric-heavy tables)
        digit_ratio = sum(c.isdigit() for c in line) / max(len(line), 1)
        if digit_ratio > digit_ratio_threshold:
            continue
        
        # Remove lines with excessive special characters
        special_chars = sum(1 for c in line if not c.isalnum() and not c.isspace())
        if special_chars / max(len(line), 1) > special_char_threshold:
            continue
        
        filtered_lines.append(line)
    
    return "\n".join(filtered_lines).strip()




#print(get_reports("JBH", year=2020, source="live_wire", ASX_200=True))
#print(get_article_text(get_reports("NAB", year=2025, source="bell_potter", ASX_200=True)[0]["url"], source="bell_potter"))


#print(get_article_text("https://www.morningstar.com.au/stocks/have-profits-peaked-for-the-big-four-banks", source="morningstar"))
#print(get_reports("NAB", source="wilsonsadvisory", ASX_200=True))

#print(clean_GPFS_text(extract_text_from_GPFS("https://cdn-api.markitdigital.com/apiman-gateway/ASX/asx-research/1.0/file/2995-01666348-2A881717&v=4a466cc3f899e00730cfbfcd5ab8940c41f474b6"[:1000])))
