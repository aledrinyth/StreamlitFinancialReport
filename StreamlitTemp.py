import streamlit as st
import pandas as pd
import requests

# --- UI Element Definitions ---

# These elements are implied by the logic but not defined in the snippet provided.
# I'm defining them here to make the UI code runnable.
st.set_page_config(layout="wide")

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

            url = "https://financescraper-jn98.onrender.com/scrape"

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