import streamlit as st
import pandas as pd
from supabase import create_client, Client
import json
from dashboard.model import analyze_report
from dashboard.utils import find_similar_companies, get_industry_companies_with_metrics
import requests

# Set the layout for the page
st.set_page_config(layout="wide") 

# Initialize Supabase client
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

# Load data from Supabase
@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_data_from_supabase(table_name: str) -> pd.DataFrame:
    try:
        response = supabase.table(table_name).select("*").execute()
        return pd.DataFrame(response.data)
    except Exception as e:
        st.error(f"Error loading {table_name}: {e}")
        return pd.DataFrame()

# Utils
def format_number(val):
    try:
        val = float(val)
        return f"{int(val):,}" if val.is_integer() else f"{val:,.2f}"
    except Exception:
        return val

# Read definitions from definitions.json
with open("dashboard/definitions.json", "r", encoding="utf-8") as f:
    definitions_data = json.load(f)
    definitions = {}
    for metric in definitions_data["metrics"]:
        definitions[metric["id"]] = metric["definition"]
        definitions[metric["name"]] = metric["definition"]
        for alias in metric.get("aliases", []):
            definitions[alias] = metric["definition"]

# Sidebar for navigation
with st.sidebar.expander("Navigation", expanded=True):
    page = st.radio("Go to", ["Financial Dashboard", "Report Analyst", "Financial Report Analyser"], label_visibility="collapsed")

# Logic of the page
if page == "Financial Dashboard":
    st.title("Company Financial Dashboard")

    # Load data from Supabase tables
    with st.spinner("Loading data from database..."):
        financial_df = load_data_from_supabase("Financial Data")
        balance_sheet_df = load_data_from_supabase("Balance Sheet")
        cash_flow_df = load_data_from_supabase("Cash Flow")
        sector_means_df = load_data_from_supabase("Sector Means")

    # Create datasets dictionary
    datasets = {
        "Financial Data": financial_df,
        "Balance Sheet": balance_sheet_df,
        "Cash Flow": cash_flow_df
    }

    # Load Ticker-to-Sector map
    ticker_to_sector = {}
    if 'ticker' in financial_df.columns and 'sector' in financial_df.columns:
        ticker_to_sector = financial_df.set_index('ticker')['sector'].to_dict()
    
    # Set sector as index for sector_means_df
    if 'sector' in sector_means_df.columns:
        sector_means_df = sector_means_df.set_index('sector')

    ticker = st.text_input("Enter ASX Ticker (e.g., NAB, CBA, ANZ):").strip().upper()

    if ticker:
        company_sector = ticker_to_sector.get(ticker)
        sector_average_row = sector_means_df.loc[company_sector] if company_sector in sector_means_df.index else None

        for section, df in datasets.items():
            st.subheader(section)

            if df.empty:
                st.warning(f"No data available for {section}")
                continue

            if ticker not in df['ticker'].values:
                st.warning(f"{ticker} not found in {section}")
                continue

            try:
                # Run find_similar_companies for this dataset
                similarity_results = find_similar_companies(df, threshold=0.1)

                row = df[df['ticker'] == ticker].iloc[0]
                cols_to_drop = ['ticker']
                if 'sector' in row.index:
                    cols_to_drop.append('sector')
                row = row.drop(cols_to_drop, errors='ignore').dropna()

                if row.empty:
                    st.info("No available data.")
                else:
                    industry_averages = []
                    industry_companies = []
                    for metric in row.index:
                        avg_value = "N/A"
                        if sector_average_row is not None and metric in sector_average_row.index:
                            avg_value = format_number(sector_average_row[metric])
                        industry_averages.append(avg_value)
                        
                        # Get industry companies with their metric values
                        industry_peers = get_industry_companies_with_metrics(df, ticker, metric, top_n=3)
                        industry_companies.append(industry_peers)

                    clean_df = pd.DataFrame({
                        "Metric": row.index,
                        "Value": [format_number(v) for v in row.values],
                        "Company that has similar metric": [similarity_results.get(ticker, {}).get(metric, "N/A") for metric in row.index],
                        "Industry Companies": industry_companies,
                        "Industry Average": industry_averages
                    })

                    # Create table layout
                    with st.container():
                        st.markdown('<div class="table-header">', unsafe_allow_html=True)
                        col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 1, 3, 3, 2])
                        with col1:
                            st.markdown('<div class="table-cell metric">Metric</div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown('<div class="table-cell value">Value</div>', unsafe_allow_html=True)
                        with col3:
                            st.markdown('<div class="table-cell notes">Notes</div>', unsafe_allow_html=True)
                        with col4:
                            st.markdown('<div class="table-cell similar">Company that has similar metric</div>', unsafe_allow_html=True)
                        with col5:
                            st.markdown('<div class="table-cell industry">Industry Companies</div>', unsafe_allow_html=True)
                        with col6:
                            st.markdown('<div class="table-cell spacer">Industry Average</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                        for idx, row in clean_df.iterrows():
                            st.markdown('<div class="table-row">', unsafe_allow_html=True)
                            col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 1, 3, 3, 2])
                            with col1:
                                st.markdown(f'<div class="table-cell metric">{row["Metric"]}</div>', unsafe_allow_html=True)
                            with col2:
                                st.markdown(f'<div class="table-cell value">{row["Value"]}</div>', unsafe_allow_html=True)
                            with col3:
                                with st.popover("View"):
                                    definition = definitions.get(row["Metric"], "No definition available")
                                    st.markdown(definition)
                            with col4:
                                st.markdown(f'<div class="table-cell similar">{row["Company that has similar metric"]}</div>', unsafe_allow_html=True)
                            with col5:
                                st.markdown(f'<div class="table-cell industry">{row["Industry Companies"]}</div>', unsafe_allow_html=True)
                            with col6:
                                st.markdown(f'<div class="table-cell industry-avg">{row["Industry Average"]}</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error loading {section}: {e}")
elif (page == "Financial Dashboard"):

    st.title("Financial Report Analyzer")

    uploaded_pdf = st.file_uploader("Upload annual report (PDF)", type="pdf")
    manual_ticker = st.text_input("Enter Ticker (e.g., MSFT)")
    run_btn = st.button("Analyze")

    # --- UI Logic from your code ---

    if run_btn:

        # --- Data Fetching UI ---
        yf_ticker = manual_ticker
        with st.spinner(f"Fetching Yahoo Finance data for {yf_ticker}..."):
            # Placeholder for Yahoo Finance data fetching
            try:
                # Simulating success or failure

                # Initialise all the parameters needed for the GPFS requests and the financial statement request
                url = st.secrets["url"]

                SUPABASE_URL = st.secrets["SUPABASE_URL"]
                SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
                FUNCTION_NAME = "Retrieve-latest-url-for-GPFS" 
                EQUITY_REPORTS = "Equity-Reports-Retrieval-ASX200"
                equity_url = f"{SUPABASE_URL}/functions/v1/{EQUITY_REPORTS}"
                invoke_url = f"{SUPABASE_URL}/functions/v1/{FUNCTION_NAME}"

                GPFS_headers = {
                    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                    "Content-Type": "application/json"
                }

                GPFS_payload = {
                    "ticker": yf_ticker,
                    "limit": 1
                }

                # --- Make the call to retrieve the financial statement data --- 
                headers = {"Content-Type": "application/json"}

                payload = {"ticker": yf_ticker}

                response = requests.post(url, headers=headers, json=payload)

                response.raise_for_status()

                data = response.json()

                # Create tabs for each financial statement
                tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])

                with tab1:
                    st.subheader("Income Statement")
                    # Extract the relevant data
                    income_data = data.get("incomeStatement")
                    if income_data:
                        # Convert the list of dictionaries to a pandas DataFrame
                        df_income = pd.DataFrame(income_data)
                        # Set the 'Breakdown' column as the index for better readability
                        df_income = df_income.set_index("Breakdown")
                        # Display the data in an interactive dataframe
                        st.dataframe(df_income)
                    else:
                        st.warning("Income Statement data not available.")

                with tab2:
                    st.subheader("Balance Sheet")
                    balance_sheet_data = data.get("balanceSheet")
                    if balance_sheet_data:
                        df_balance = pd.DataFrame(balance_sheet_data)
                        df_balance = df_balance.set_index("Breakdown")
                        st.dataframe(df_balance)
                    else:
                        st.warning("Balance Sheet data not available.")

                with tab3:
                    st.subheader("Cash Flow")
                    cash_flow_data = data.get("cashFlow")
                    if cash_flow_data:
                        df_cash_flow = pd.DataFrame(cash_flow_data)
                        df_cash_flow = df_cash_flow.set_index("Breakdown")
                        st.dataframe(df_cash_flow)
                    else:
                        st.warning("Cash Flow data not available.")

                # --- Now retrieve the GPFS from our database ---
                try:
                    st.markdown("General Purpose Financial Statements(GPFS)")

                    GPFS_payload = {
                        "ticker": yf_ticker.split('.')[0],
                        "limit": 1
                    }
                    
                    GPFS_response = requests.post(invoke_url, headers=GPFS_headers, json=GPFS_payload, timeout=10)
        
                    # Raise an exception for bad status codes (4xx or 5xx)
                    GPFS_response.raise_for_status()
                
                    # --- Process the Response ---
                    
                    # The response from the function is in JSON format
                    GPFS_data = GPFS_response.json()
                    url = None  # Default value if not found

                    processed_items = set()

                    # Check if 'data' key exists and if the list is not empty
                    for item in GPFS_data.get('data', []):
                        year = item.get('year')
                        url = item.get('url')
                    
                        # --- Robustness Check ---
                        # Skip this iteration if 'year' or 'url' is missing. [11, 13]
                        if not year or not url:
                            continue
                    
                        # --- Main Logic ---
                        # Check if the year is within the desired range.
                        if 2020 <= year < 2025:
                            # Create a unique identifier for the current item.
                            item_identifier = (year, url)
                    
                            # --- Duplicate Check ---
                            # If this combination has not been processed yet, display it.
                            if item_identifier not in processed_items:
                                st.markdown(f"Check out this for {year} [link]({url})")
                    
                                # Add the identifier to the set to prevent future duplicates.
                                processed_items.add(item_identifier)
                except Exception as f:
                    st.error(f"GPFS retrieval error {f}")

                try:
                    st.markdown("Equity Reports")

                    Equity_payload = {
                        "ticker": yf_ticker.split('.')[0],
                        "limit": 1
                    }
                    
                    equity_response = requests.post(equity_url, headers=GPFS_headers, json=Equity_payload, timeout=10)
        
                    # Raise an exception for bad status codes (4xx or 5xx)
                    equity_response.raise_for_status()
                
                    # --- Process the Response ---
                    
                    # The response from the function is in JSON format
                    equity_data = equity_response.json()
                    url = None  # Default value if not found

                    processed_items = set()

                    # Check if 'data' key exists and if the list is not empty

                    counter = 1
                    
                    for item in equity_data.get('data', []):
                        url = item.get('url')
                        st.markdown(f"{counter}) Check this out [link]({url})")
                        counter+=1
                    
                except Exception as f:
                    st.error(f"Equity Report retrieval error {f}")


            except Exception as e:
                st.error(f"Yahoo Finance error: {e}")




    if 'report' not in st.session_state:
        st.info("Upload a PDF or enter an override ticker, then click **Analyze**.")
    else:
        # Placeholder for the actual render_report(report) function
        # st.header(f"Analysis for {st.session_state['report']['metadata']['ticker']}")
        st.success("Report generated successfully!")
        st.write("Report content would be displayed here.")
        st.balloons()
