import pandas as pd
import numpy as np

def format_number(val):
    """
    Format numeric values for display, handling both strings and numbers.
    """
    try:
        val = float(val)
        return f"{int(val):,}" if val.is_integer() else f"{val:,.2f}"
    except Exception:
        return str(val)

def find_similar_companies(data, threshold=0.1, skip_cols=None, max_results=3):
    """
    For each company (row, indexed by Ticker) and each numeric metric column (excluding skipped columns),
    find other companies with values within a relative threshold.
    
    Parameters:
    - data: pandas DataFrame (companies in 'ticker' column, metrics in other columns)
    - threshold: relative difference to consider "similar" (default 10%)
    - skip_cols: list of column names to ignore (e.g., ['ticker']). Defaults to ['ticker']
    
    Returns:
    - dict: {company: {metric: "similar_str" or "N/A" or "None"}}
    """
    if skip_cols is None:
        skip_cols = ['ticker']
    
    # Set 'ticker' as index if it exists
    if 'ticker' not in data.columns:
        raise ValueError("Column 'ticker' not found in DataFrame")
    data = data.set_index('ticker')
    
    # Keep only numeric columns that are not in skip_cols
    numeric_cols = [
        col for col in data.columns
        if np.issubdtype(data[col].dtype, np.number) and col not in skip_cols
    ]
    
    companies = data.index.values
    result = {company: {} for company in companies}
    
    for company in companies:
        company_row = data.loc[company]
        
        for metric in numeric_cols:
            value = company_row[metric]
            
            # Handle missing / zero values
            if pd.isna(value) or value == 0:
                result[company][metric] = "N/A"
                continue
            
            similar = []
            for other in companies:
                if other == company:
                    continue
                oth_val = data.loc[other, metric]
                
                if pd.isna(oth_val) or oth_val == 0:
                    continue
                
                # Relative similarity check
                if abs(value - oth_val) / abs(value) <= threshold:
                    similar.append(f"{other} ({oth_val:.2f})")
            similar = similar[:max_results]
            result[company][metric] = "\n".join(similar) if similar else "None"
    
    return result

def get_industry_companies_with_metrics(data, target_company, metric_col, top_n=3):
    """
    Find companies in the same industry and return their values for a specific metric.
    
    Parameters:
    - data: pandas DataFrame with 'ticker' and 'industry' columns
    - target_company: ticker of the company to find peers for
    - metric_col: the specific metric column to get values for
    - top_n: number of companies to return (default 3)
    
    Returns:
    - str: formatted string with company tickers and their metric values, or "N/A"
    """
    
    # Find target company's industry
    target_row = data[data['ticker'] == target_company]
    if target_row.empty:
        return "N/A"
    
    target_industry = target_row['industry'].iloc[0]
    
    # Find other companies in same industry with non-null values for the metric
    same_industry = data[
        (data['industry'] == target_industry) & 
        (data['ticker'] != target_company) &
        (data[metric_col].notna())
    ]
    
    if same_industry.empty:
        return "None"
    
    # Convert to numeric for sorting, but keep original values for display
    same_industry = same_industry.copy()
    numeric_col = pd.to_numeric(same_industry[metric_col], errors='coerce')
    same_industry['_numeric_sort'] = numeric_col
    
    # Sort by numeric value (descending) and take top_n
    same_industry = same_industry.dropna(subset=['_numeric_sort']).sort_values(by='_numeric_sort', ascending=False).head(top_n)
    
    # Format results
    results = []
    for _, row in same_industry.iterrows():
        value = row[metric_col]
        if pd.isna(value):
            continue
        
        # Use format_number function to handle formatting consistently
        formatted_value = format_number(value)
        results.append(f"{row['ticker']} ({formatted_value})")
    
    return "\n".join(results) if results else "None"