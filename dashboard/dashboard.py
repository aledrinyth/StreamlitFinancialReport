import streamlit as st
import pandas as pd
from supabase import create_client, Client
import json
from model import analyze_report
from utils import find_similar_companies, get_industry_companies_with_metrics
import requests
from ../using.sentence_model import process_single_document, results_to_dataframe
from ../using.text_extracter import get_reports, get_article_text, get_GPFS_reports, extract_text_from_GPFS, clean_GPFS_text

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
elif (page == "Financial Report Analyser"):

    st.title("Financial Report Analyser")

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

elif (page == "Equity Report Analyser"):
    
    
    
    # Motley fool = data base error
    # Buy hold sell =  either database error or not saved as BHS
    # Livewire = cannot be scraped need to change scraper
    # Money of mine = cloudscraper needed
    # Morningstar = works but need to add rest period not to overwhelm server
    
    
    
    
    
    report_sources = {
        "Bell Potter": "bell_potter",
        "Buy Hold Sell": "buy_hold_sell",
        "Motley Fool": "motel_fool",
        "Livewire": "live_wire",
        "Money of Mine": "money_of_mine",
        "Morningstar": "morningstar",
        "Ord Minnett": "ord_minnet",
        "Wilson Advisory": "wilsonsadvisory",
    }
    def get_key_by_value(d, value):
        for k, v in d.items():
            if v == value:
                return k
        return None  # if not found
    
    
    
    st.set_page_config(page_title='Equity Reports Sentiment Analyser Dashboard', layout ='wide')
    st.title("Equity Reports Sentiment Analyser Dashboard")
    
    st.markdown("This dashboard allows you to analyse sentiment of equity reports using FinBERT model.")
    
    
    tab_pre_scraped, tab_new_reports = st.tabs([
        "📚 Pre-scraped Reports",
        "🌐 Scrape New Report"
    ])
    
    
    def process_next_report_batch(batch_size=10):
        start = st.session_state.current_report_index
        end = min(start + batch_size, len(st.session_state.all_reports))
    
        bar = st.progress(0)
    
        for i in range(start, end):
            report = st.session_state.all_reports[i]
    
            report_text = get_article_text(report["url"], source=report["source"])
            report_sentiment = process_single_document(text=report_text)
    
            report["sentiment"] = {
                "neg": report_sentiment["agg_probs"][0],
                "neu": report_sentiment["agg_probs"][1],
                "pos": report_sentiment["agg_probs"][2],
            }
    
            st.session_state.processed_report_results.append(
                {
                    "year": report["year"],
                    "source": get_key_by_value(report_sources, report["source"]),
                    "ticker": report["ticker"],
                    "link": report["url"],
                    "industry": report["industry"],
                    "team_industry": report["investment_team_industry"],
                    "sentiment": report["sentiment"],
                }
            )
    
            bar.progress((i - start + 1) / (end - start))
            time.sleep(2) # Add buffer to avoid overwhelming the server and to simulate processing time
    
        st.session_state.current_report_index = end
    
    def process_next_batch_GPFS(batch_size=10):
        start = st.session_state.current_index_GPFS
        end = min(start + batch_size, len(st.session_state.all_GPFS_reports))
    
        bar = st.progress(0, text="Processing GPFS reports...")
    
        for i in range(start, end):
            report = st.session_state.all_GPFS_reports[i]
    
            # Assuming you have a similar function to extract GPFS text
            report_text = extract_text_from_GPFS(report["url"])
            cleaned_text = clean_GPFS_text(report_text)
            report_sentiment = process_single_document(text=cleaned_text)
    
            report["sentiment"] = {
                "neg": report_sentiment["agg_probs"][0],
                "neu": report_sentiment["agg_probs"][1],
                "pos": report_sentiment["agg_probs"][2],
            }
    
            st.session_state.processed_GPFS_results.append(
                {
                    "year": report["year"],
                    "ticker": report["ticker"],
                    "link": report["url"],
                    "sentiment": report["sentiment"],
                }
            )
    
            bar.progress((i - start + 1) / (end - start))
            time.sleep(2)
    
        st.session_state.current_index_GPFS = end
    
    
    
    with tab_pre_scraped:
        st.header("Pre-scraped Equity Reports Analysis")
        st.markdown("Select from the pre-scraped equity reports to view sentiment analysis results.")
        st.markdown("The following options will allow you to narrow down the reports to analyze. The ticker option is compulsory, and at least one of the year and the source must be selected.")
        
    
        ticker = st.text_input('Select a Ticker, e.g., CBA, BHP, TLS')
        ticker_clean = ticker.strip().upper() if ticker else ""
        ASX_200 = st.checkbox('Is the ticker part of ASX 200?', value=True)
    
    
        year = st.multiselect('Select Year, e.g., 2023, 2022 (optional)', options=list(range(2026, 2019, -1)))
        year_selected_flag = len(year) > 0
    
    
        st.warning('As of right now only the following sources are supported: Bell Potter, Wilson Advisory, Ord Minnett, Morningstar.')
        selected_label = st.multiselect(
        "Select report source:",
        list(report_sources.keys()))
        selected_source_value = [report_sources[label] for label in selected_label]
        source_selected_flag = len(selected_source_value) > 0
    
    
    
    
        # change to normal button state management once loading bar option is choosen
        if "analyze_clicked" not in st.session_state:
            st.session_state.analyze_clicked = False
    
        if "analysis_results" not in st.session_state:
            st.session_state.analysis_results = []
    
        if "num_results_to_show" not in st.session_state:
            st.session_state.num_results_to_show = 5
        
        if "all_reports" not in st.session_state:
            st.session_state.all_reports = []
        
        if "all_GPFS_reports" not in st.session_state:
            st.session_state.all_GPFS_reports = []
    
        if "processed_report_results" not in st.session_state:
            st.session_state.processed_report_results = []
    
        if "current_report_index" not in st.session_state:
            st.session_state.current_report_index = 0
    
        if 'current_index_GPFS' not in st.session_state:
            st.session_state.current_index_GPFS = 0
    
        if 'processed_GPFS_results' not in st.session_state:    
            st.session_state.processed_GPFS_results = []
    
    
    
        if st.button("Analyze Selected Report"):
            st.session_state.analyze_clicked = True
            st.session_state.num_results_to_show = 5  # reset pagination
            st.session_state.processed_report_results = []
            st.session_state.current_report_index = 0
    
    
            # Validation
            if not ticker_clean:
                st.error("Ticker is required!")
            elif not (year_selected_flag or source_selected_flag):
                st.error("Please select at least one of Year or Source.")
            else:
                # Passed validation
                st.write(f"Analyzing sentiment for Ticker: {ticker_clean}")
    
                # show selected filters
                st.write(f"Year(s): {year if year_selected_flag else 'All'}")
                st.write(f"Source(s): {', '.join(selected_label) if source_selected_flag else 'All'}")
    
                reports = get_reports(ticker_clean, year=year if year_selected_flag else None, source=selected_source_value if source_selected_flag else None, ASX_200=ASX_200)
                reports = sorted(reports, key=lambda x: (-x["year"], x["source"]))
                GPFS_reports = get_GPFS_reports(ticker_clean, year=year if year_selected_flag else None)
                GPFS_reports = sorted(GPFS_reports, key=lambda x: (-x["year"]))
    
                if reports and GPFS_reports:
                    st.write(f"Found {len(reports)} equity report(s) matching the criteria.")
                    st.write(f"Found {len(GPFS_reports)} GPFS report(s) matching the criteria.")
                    st.session_state.all_reports = reports
                    st.session_state.all_GPFS_reports = GPFS_reports
                elif reports:
                    st.write(f"Found {len(reports)} equity report(s) matching the criteria.")
                    st.write("No GPFS reports found for the selected criteria.")
                    st.session_state.all_reports = reports
                    st.session_state.all_GPFS_reports = []
                elif GPFS_reports:
                    st.write(f"Found {len(GPFS_reports)} GPFS report(s) matching the criteria.")
                    st.write("No equity reports found for the selected criteria.")
                    st.session_state.all_reports = []
                    st.session_state.all_GPFS_reports = GPFS_reports
                else:
                    st.warning("No reports found for the selected criteria.")
                    st.session_state.all_reports = []
                    st.session_state.all_GPFS_reports = []
        
        
        if st.session_state.analyze_clicked and (st.session_state.all_reports or st.session_state.all_GPFS_reports):
            # Process first batch automatically if nothing processed yet
            if st.session_state.current_report_index == 0 and st.session_state.all_reports:
                process_next_report_batch(10)
            if st.session_state.current_index_GPFS == 0 and st.session_state.all_GPFS_reports:
                process_next_batch_GPFS(10)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.header("Equity Report Sentiment Analysis Results")
    
                for report in st.session_state.processed_report_results:
                    with st.expander(f"{report['year']} - {report['source']} - {report['ticker']}"):
                        st.markdown(f"[Original Report]({report['link']})")
                        st.write(f"Industry: {report['industry']}")
                        st.write(f"Investment Team Industry: {report['team_industry']}")
                        st.write(
                            f"Sentiment: Pos: {report['sentiment']['pos']*100:.1f}%, "
                            f"Neu: {report['sentiment']['neu']*100:.1f}%, "
                            f"Neg: {report['sentiment']['neg']*100:.1f}%"
                        )
    
    
                # Show more button
                if st.session_state.current_report_index < len(st.session_state.all_reports):
                    if st.button("Show more"):
                        process_next_report_batch(10)
            
            with col2:
                st.header("GPFS Report Sentiment Analysis Results")
    
                for report in st.session_state.processed_GPFS_results:
                    with st.expander(f"{report['year']} - {report['ticker']}"):
                        st.markdown(f"[Original Report]({report['link']})")
                        st.write(
                            f"Sentiment: Pos: {report['sentiment']['pos']*100:.1f}%, "
                            f"Neu: {report['sentiment']['neu']*100:.1f}%, "
                            f"Neg: {report['sentiment']['neg']*100:.1f}%"
                        )
    
                # Show more button
                if st.session_state.current_index_GPFS < len(st.session_state.all_GPFS_reports):
                    if st.button("Show more GPFS"):
                        process_next_batch_GPFS(10)
                    
            """
            results_to_show = st.session_state.analysis_results[:st.session_state.num_results_to_show]
    
            for report in results_to_show:
                with st.expander(f"{report['year']} - {report['source']} - {report['ticker']}"):
                    st.markdown(f"[Original Report]({report['link']})")
                    st.write(f"Industry: {report['industry']}")
                    st.write(f"Investment Team Industry: {report['team_industry']}")
                    st.write(
                        f"Sentiment: Pos: {report['sentiment']['pos']*100:.1f}%, "
                        f"Neu: {report['sentiment']['neu']*100:.1f}%, "
                        f"Neg: {report['sentiment']['neg']*100:.1f}%"
                    )
    
            # Show more button
            if st.session_state.num_results_to_show < len(st.session_state.analysis_results):
                if st.button("Show more", key="show_more"):
                    st.session_state.num_results_to_show += 10
            """
    
    
    
    with tab_new_reports:
        st.header("Scrape and Analyze New Equity Report")
        st.markdown("Input a URL to scrape a new equity report and analyze its sentiment.")
    
        report_url = st.text_input('Enter the URL of the equity report to scrape and analyze:')
        if "scrape_done" not in st.session_state:
                st.session_state.scrape_done = False
    
        # Scrape button
        if st.button("Scrape and Analyze"):
    
            if not report_url.strip():
                st.error("Please enter a valid URL.")
            else:
                st.write(f"Scraping and analyzing report from URL: {report_url.strip()}")
    
                with st.spinner(text="Scraping and analyzing..."):
                    time.sleep(5)  
    
                st.success("Scraping and sentiment analysis completed!")
                st.write("Sentiment Results:")
                st.write("Positive: 65%")
                st.write("Neutral: 25%")
                st.write("Negative: 10%")
    
                
                st.session_state.scrape_done = True
    
        # Only show additional inputs if scraping is done
        if st.session_state.scrape_done:
            st.markdown("Could you please fill in the following information to add this report to the database for future use.")
    
            
            ticker_input = st.text_input('Select the relevant tickers separated by a space, e.g. CBA BHP TLS')
            tickers = [t.upper() for t in ticker_input.strip().split() if t.strip()]
    
            year = st.number_input('Select Year, e.g., 2023, 2022', min_value=2000, max_value=2026, step=1)
    
            selected_label = st.text_input("Select the report source:")
            clean_label = selected_label.strip().lower().replace(" ", "_")
    
    
            st.write("Given inputs:")
            st.write(f"Ticker(s): {', '.join(tickers)}")
            st.write(f"Year: {year}")
            st.write(f"Source: {clean_label}")



