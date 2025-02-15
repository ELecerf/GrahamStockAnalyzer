#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 19:54:04 2025

@author: elecerf
"""

import streamlit as st
import pandas as pd
import requests
from pymongo import MongoClient
import datetime
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
import plotly.express as px
import country_converter as coco
import streamlit.components.v1 as components
import numpy as np
import logging
from typing import Tuple, Any, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="ValeurGraph", page_icon="📈")

# =============================================================================
# API Keys (Use st.secrets for sensitive info)
# =============================================================================
FINNHUB_API_KEY = 'your_finnhub_api_key'
EOD_API_KEY = st.secrets["EOD_API_KEY"]
MONGO_DB = st.secrets["MONGO_DB"]
GUMROAD_API_URL = st.secrets["GUMROAD_API_URL"]
PRODUCT_ID = st.secrets["PRODUCT_ID"]

# =============================================================================
# MongoDB Connection & Basic Data Functions
# =============================================================================
client = MongoClient(MONGO_DB)
db = client.valeurgraphDB
Collection = db["data"]

@st.cache_data
def load_data(exchanges=['TSE']):
    with st.spinner('Loading data'):
        query = {'Exchange': {'$in': exchanges}} if exchanges else {}
        cursor = Collection.find(query)
        df = pd.DataFrame(list(cursor))
        columns = [
            'Name', 'Exchange', 'Code', 'close', 'GrahamNumberToPrice', 
            'NCAV_0toMarketCap', 'Net_Cash_to_MarketCap', 'NCAV_1toMarketCap',
            'Enterprising_Number', 'Criteria_Def2A', 'Criteria_Def2B', 'DilutedEpsTTM', 'Country'
        ]
        return df[columns]

@st.cache_data
def load_data_NCAV():
    with st.spinner('Loading data'):
        query = {'NCAV_0toMarketCap': {'$gt': 100}}
        cursor = Collection.find(query)
        df = pd.DataFrame(list(cursor))
        columns = ['Name', 'Exchange', 'Code', 'close', 'GrahamNumberToPrice', 'NCAV_0toMarketCap', 'Country']
        return df[columns]

@st.cache_data
def stocks_per_country(df):
    country_stock_count = df.groupby('Country').size().reset_index(name='Net-Nets')
    cc = coco.CountryConverter()
    some_names = country_stock_count["Country"]
    country_stock_count["CountryISO"] = cc.convert(names=some_names, to='ISO3')
    return country_stock_count

def netnetmap(df):
    fig = px.choropleth(
        df,
        locations="CountryISO",
        color="Net-Nets",
        hover_name="Country",
        hover_data={"Net-Nets": True, "CountryISO": False},
        color_continuous_scale=px.colors.sequential.Plasma
    )
    fig.update_layout(
        dragmode=False,
        geo=dict(fitbounds="locations", visible=True),
        coloraxis_colorbar=dict(title="Net-Nets")
    )
    return fig

# =============================================================================
# License Check Function
# =============================================================================
def check_license(key):
    if key == 'daubasses':
        return True, None, None
    params = {"product_id": PRODUCT_ID, "license_key": key}
    try:
        response = requests.post(GUMROAD_API_URL, data=params)
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        logger.error(f"License check error: {e}")
        return False, None, None
    if result.get("success", False):
        purchase_date_str = result.get("purchase", {}).get("created_at")
        if purchase_date_str:
            purchase_date = datetime.datetime.strptime(purchase_date_str, "%Y-%m-%dT%H:%M:%SZ")
            expiration_date = purchase_date + datetime.timedelta(days=30)
            if datetime.datetime.now() <= expiration_date:
                remaining_days = (expiration_date - datetime.datetime.now()).days
                return True, remaining_days, expiration_date
            else:
                return False, None, expiration_date
    return False, None, None

# =============================================================================
# Data Fetching & Calculation (Improved Version)
# =============================================================================
def fetch_financials_with_country(ticker):
    """
    Fetches comprehensive financial data for the given ticker.
    Returns:
      - financials: DataFrame combining balance sheet and income statement data (most recent 16 records).
      - country: Company's country (string).
      - dividends: Dividend history DataFrame with columns ['Year', 'Count'].
    Raises:
      - ValueError if data is not available.
    """
    url = f"https://eodhistoricaldata.com/api/fundamentals/{ticker}"
    params = {'api_token': EOD_API_KEY, 'fmt': 'json', 'from': '2005-01-01'}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        raise ValueError(f"Error fetching financial data for {ticker}: {e}")
    
    if not data:
        raise ValueError(f"Financial data for ticker {ticker} is not available")
    
    # Process Balance Sheet Data
    balance_sheet = data.get('Financials', {}).get('Balance_Sheet', {})
    columns_of_interest = [
        'commonStockSharesOutstanding', 'totalAssets', 'intangibleAssets', 'goodWill', 
        'totalLiab', 'totalCurrentAssets', 'totalCurrentLiabilities', 'longTermDebt', 
        'netTangibleAssets', 'cash', 'totalStockholderEquity', 'nonCurrentLiabilitiesTotal', 
        'propertyPlantAndEquipmentNet'
    ]
    try:
        df_yearly_bs = pd.DataFrame.from_dict(balance_sheet.get('yearly', {}), orient="index")[columns_of_interest]
        df_quarterly_bs = pd.DataFrame.from_dict(balance_sheet.get('quarterly', {}), orient="index")[columns_of_interest]
    except Exception as e:
        raise ValueError(f"Error processing balance sheet data for {ticker}: {e}")
    
    df_yearly_bs.index = pd.to_datetime(df_yearly_bs.index)
    df_quarterly_bs.index = pd.to_datetime(df_quarterly_bs.index)
    balance_sheet_df = pd.concat([df_quarterly_bs[:3], df_yearly_bs]).sort_index(ascending=False)
    
    # Process Income Statement Data
    income_statement = data.get('Financials', {}).get('Income_Statement', {})
    income_columns = ['totalRevenue', 'netIncome']
    try:
        df_yearly_is = pd.DataFrame.from_dict(income_statement.get('yearly', {}), orient="index")[income_columns]
    except Exception as e:
        raise ValueError(f"Error processing income statement data for {ticker}: {e}")
    
    df_yearly_is.index = pd.to_datetime(df_yearly_is.index)
    income_statement_df = df_yearly_is.sort_index(ascending=False)
    
    # Process Dividend History
    dividends_data = data.get('SplitsDividends', {}).get('NumberDividendsByYear', {})
    cols = ['Year', 'Count']
    if dividends_data:
        dividends = pd.DataFrame.from_dict(dividends_data, orient="index", columns=cols)
    else:
        dividends = pd.DataFrame(columns=cols)
    
    # Extract Country Information
    country = data.get('General', {}).get('AddressData', {}).get('Country', 'Unknown')
    
    # Extract highlights Information
    diluted_eps_ttm = round(data.get('Highlights', {}).get('DilutedEpsTTM', np.nan),2)
    
    # Combine Financial Data
    financials = balance_sheet_df.join(income_statement_df, how='outer').drop_duplicates()
    financials.index = pd.to_datetime(financials.index)
    financials = financials.sort_index(ascending=False)
    financials = financials.apply(pd.to_numeric, errors='coerce')
    financials = financials[:16]
    
    return financials, country, dividends, diluted_eps_ttm

def calculate_values(df):
    """
    Calculates derived per-share financial metrics.
    """
    df['Cash'] = round(df['cash'] / df['commonStockSharesOutstanding'], 2)
    df['NCAV'] = round((df['totalCurrentAssets'] - df['totalLiab']) / df['commonStockSharesOutstanding'], 2)
    df['NTAV'] = round((df['totalCurrentAssets'] + df['propertyPlantAndEquipmentNet']- df['totalLiab']) / df['commonStockSharesOutstanding'], 2)
    df['BookValue'] = round(df['totalStockholderEquity'], 2)
    df['BookValuePerShare'] = round(df['BookValue'] / df['commonStockSharesOutstanding'], 2)
    df['EPS'] = round(df['netIncome'] / df['commonStockSharesOutstanding'], 2)
    df['10EPS'] = round(df['netIncome'] * 10 / df['commonStockSharesOutstanding'], 2)
    df['EPS3'] = round(df['EPS'][::-1].rolling(window=3, min_periods=1).mean(),2)[::-1]
    df['Graham_Number'] = round((22.5 * df['BookValuePerShare'].clip(0) * df['EPS3'].clip(0)) ** 0.5, 2)
    df['Graham_Number_Entp'] = round((12 * df['NTAV'].clip(0) * df['EPS'].clip(0)) ** 0.5, 2)        
    df['AnnualSales'] = round(df['totalRevenue'], 2)
    df['BV%'] = round(df['BookValue'] / df['totalAssets'] * 100, 1)
    df['Liab%'] = round(df['totalLiab'] / df['totalAssets'] * 100, 1)
    df['Current Assets/2*Current Liab'] = round(100 * df['totalCurrentAssets'] / (2 * df['totalCurrentLiabilities']), 2)
    df['Current Assets'] = df['totalCurrentAssets']
    df['Net Current Asset/Non Current Liabilities'] = round((df['totalCurrentAssets'] - df['totalLiab']) / df['nonCurrentLiabilitiesTotal'], 2)
    df['2*equity/debt'] = round(2 * 100 * df['totalAssets'] / df['totalLiab'])
        
    return df

class DataProcessor:
    @staticmethod
    def calculate_values(df):
        return calculate_values(df)

def get_fundamentals(tick):
    """
    Retrieves combined financial data using fetch_financials_with_country,
    processes it to calculate derived metrics, and returns a subset based on license status.
    """
    try:
        financials, country, dividends, diluted_eps_ttm = fetch_financials_with_country(tick)
    except Exception as e:
        st.error(f"Error fetching financials for {tick}: {e}")
        return pd.DataFrame()
    
    try:
        financials = DataProcessor.calculate_values(financials.astype(float))
    except Exception as e:
        st.error(f"Error processing financial data for {tick}: {e}")
        return pd.DataFrame()
    
    if st.session_state.get('license_valid', False):
        selected_metrics = [
            'Graham_Number', 'NCAV', 'Cash', '10EPS', 'AnnualSales', 'netIncome',
            'NTAV','BookValuePerShare', 'Current Assets/2*Current Liab',
            'Current Assets', 'Net Current Asset/Non Current Liabilities'
        ]
    else:
        selected_metrics = ['BookValuePerShare']
    
    return financials, country, dividends, diluted_eps_ttm

def get_full_fundamentals(tick):
    """
    Retrieves and processes the complete financials DataFrame without license-based filtering.
    This is used for plotting so that all necessary metrics are available.
    """
    try:
        financials, country, dividends, diluted_eps_ttm = fetch_financials_with_country(tick)
    except Exception as e:
        st.error(f"Error fetching financials for {tick}: {e}")
        return pd.DataFrame()
    
    try:
        financials = DataProcessor.calculate_values(financials.astype(float))
    except Exception as e:
        st.error(f"Error processing financial data for {tick}: {e}")
        return pd.DataFrame()
    
    return financials

def get_price_eod(tick):
    """
    Retrieves historical adjusted close prices for the given ticker.
    """
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=4000)
    url = f"https://eodhistoricaldata.com/api/eod/{tick}"
    params = {
        'api_token': EOD_API_KEY,
        'from': start.strftime('%Y-%m-%d'),
        'to': end.strftime('%Y-%m-%d'),
        'fmt': 'json'
    }
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        price = pd.DataFrame.from_dict(r.json())
        price = price.set_index('date')
        price.index = pd.to_datetime(price.index)
        price = price[['adjusted_close']]
        return price
    except Exception as e:
        st.error(f"Error fetching price data for {tick}: {e}")
        return pd.DataFrame()

# =============================================================================
# UI Functions (Search, Screener, Plotting, etc.)
# =============================================================================
@st.cache_data
def search_stocks(query):
    if query:
        url = f'https://eodhistoricaldata.com/api/search/{query}'
        params = {'api_token': EOD_API_KEY, 'type': 'stock', 'limit': '30'}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            js = response.json()
            query_df = pd.DataFrame(js)
            return query_df
        else:
            st.error('Failed to fetch data from EOD Historical Data API')
            return pd.DataFrame()
    else:
        return pd.DataFrame()

def search_command():
    with st.form("Search Form"):
        st.title("Search company")
        query = st.text_input("Enter a stock symbol or name to search:", "")
        search_button = st.form_submit_button("Search")
    if 'result_df' not in st.session_state:
        st.session_state['result_df'] = pd.DataFrame()
    if search_button:
        if query:
            with st.spinner("Searching for stocks..."):
                st.session_state['result_df'] = search_stocks(query)
        else:
            st.info("Please enter a query to search for stocks.")
            st.session_state['result_df'] = pd.DataFrame()
    result_df = st.session_state['result_df']
    if not result_df.empty:
        st.write("Search Results:")
        with st.form("Plot Form"):
            selected_rows = st.dataframe(
                result_df,
                use_container_width=False,
                hide_index=False,
                selection_mode='single-row',
                on_select='rerun',
                key='dataframeSearch'
            )
            plot_button = st.form_submit_button("Plot selection")
            if plot_button:
                if selected_rows and 'rows' in selected_rows.selection:
                    if selected_rows.selection['rows']:
                        selected_index = selected_rows.selection['rows'][0]
                        selected_row = result_df.iloc[selected_index]
                        ticker = f"{selected_row['Code']}.{selected_row['Exchange']}"
                        st.session_state['selected_ticker'] = ticker
                        st.session_state['trigger_plot'] = True
                        st.write(f"Selected: {ticker}, see value graph below")
                    else:
                        st.write("No row selected")
                else:
                    st.write("Selection data not available")
    elif query:
        st.info("No result")

def display_screener():
    exchanges = sorted(['PA', 'XETRA', 'TSE', 'TO', 'MI', 'MC', 'US', 'AS', 'BR', 'WAR', 'OL', 'CO', 'ST'])
    with st.form("Exchange Selector"):
        st.title('Screener')
        selected_exchange = st.selectbox('Select an exchange:', exchanges)
        load_data_button = st.form_submit_button("Load Data")
    if load_data_button:
        df = load_data([selected_exchange])
        st.session_state['df'] = df
    if 'df' in st.session_state:
        df = st.session_state['df']
        with st.form("Row Selector"):
            selected_rows = st.dataframe(
                df,
                use_container_width=False,
                hide_index=False,
                selection_mode='single-row',
                on_select='rerun',
                key='dataframeScreener'
            )
            plot_button = st.form_submit_button("Plot selection")
            if plot_button:
                if selected_rows and 'rows' in selected_rows.selection:
                    if selected_rows.selection['rows']:
                        selected_index = selected_rows.selection['rows'][0]
                        selected_row = df.iloc[selected_index]
                        ticker = f"{selected_row['Code']}.{selected_row['Exchange']}"
                        st.session_state['selected_ticker'] = ticker
                        st.session_state['trigger_plot'] = True
                        st.write(f"Selected: {ticker}")
                    else:
                        st.write("No row selected")
                else:
                    st.write("Selection data not available")

def evaluate_company(data, price):
    score = 0
    score_details = {
        'Current Assets > 2*Current Liab': 0,
        'Net Current Asset > Non Current Liabilities': 0,
        'Positive Earnings for 10 Years': 0,
        'Price < NCAV': 0,
        'Price < 15 EPS': 0,
        'Price < 1.5 Book Value Per Share': 0,
    }
    if not data.empty:
        if data.iloc[0]['Current Assets/2*Current Liab'] >= 100:
            score += 1
        score_details['Current Assets > 2*Current Liab'] = round(data.iloc[0]['Current Assets/2*Current Liab'] / 100, 2)
    cleaned_eps = data['10EPS'].dropna()
    if not cleaned_eps.empty and cleaned_eps.min() >= 0:
        score += 1
        score_details['Positive Earnings for 10 Years'] = 'True'
    else:
        score_details['Positive Earnings for 10 Years'] = 'False'
    if not data.empty and not price.empty:
        if data.iloc[0]['NCAV'] / price.iloc[-1]['adjusted_close'] >= 1:
            score += 1
        score_details['Price < NCAV'] = round(data.iloc[0]['NCAV'] / price.iloc[-1]['adjusted_close'], 2)
        if data.iloc[0]['Net Current Asset/Non Current Liabilities'] >= 100:
            score += 1
        score_details['Net Current Asset > Non Current Liabilities'] = round(data.iloc[0]['Net Current Asset/Non Current Liabilities'] / 100, 2)
        if not cleaned_eps.empty and price.iloc[-1]['adjusted_close'] / (cleaned_eps.iloc[0] / 10) <= 15:
            score += 1
        score_details['Price < 15 EPS'] = round(price.iloc[-1]['adjusted_close'] / (cleaned_eps.iloc[0] / 10), 2)
        if price.iloc[-1]['adjusted_close'] / data.iloc[0]['BookValuePerShare'] <= 1.5:
            score += 1
        score_details['Price < 1.5 Book Value Per Share'] = round(price.iloc[-1]['adjusted_close'] / data.iloc[0]['BookValuePerShare'], 2)
    st.markdown("## Graham scoring")
    for criterion, criterion_score in score_details.items():
        st.markdown(f"**{criterion}:** {criterion_score}")
    st.markdown(f"**Total Score: {score}/6**")
    return score

def create_bokeh_chart(stock, df_fundamentals, df_stock):
    # Ensure the index is named "date" in both DataFrames
    df_fundamentals.index.name = "date"
    df_stock.index.name = "date"
    
    # Reset index so that 'date' becomes an explicit column
    df_fundamentals = df_fundamentals.reset_index()
    df_stock = df_stock.reset_index()
    
    # Calculate moving averages for the price data
    df_stock['10_MA'] = df_stock['adjusted_close'].rolling(window=10).mean()
    df_stock['30_MA'] = df_stock['adjusted_close'].rolling(window=30).mean()
    
    s1 = ColumnDataSource(df_fundamentals)
    s2 = ColumnDataSource(df_stock)
    
    p = figure(title=stock, x_axis_type='datetime', height=400, sizing_mode='stretch_width')
    # Plot the stock price and moving averages
    p.line('date', 'adjusted_close', source=s2, legend_label='Close Price', color='midnightblue', alpha=0.8)
    p.line('date', '10_MA', source=s2, legend_label='10-Day MA', color='orange', alpha=0.8)
    p.line('date', '30_MA', source=s2, legend_label='30-Day MA', color='green', alpha=0.8)
    
    # Plot fundamentals markers (ensure these columns exist in the fundamentals DataFrame)
    p.scatter('date', 'BookValuePerShare', source=s1, legend_label='Book Value per share', color='red', marker='circle')
    p.scatter('date', 'NCAV', source=s1, legend_label='Net Current Asset Value per share', color='blue', size=10, marker='y')
    p.scatter('date', 'Graham_Number', source=s1, legend_label='Graham Number', color='green', marker='circle')
    p.scatter('date', 'NTAV', source=s1, legend_label='Net Tangible Asset Value per share', color='black', marker='circle')
    p.scatter('date', 'Cash', source=s1, legend_label='Cash per share', color='black', size=10, marker='x')
    p.scatter('date', '10EPS', source=s1, legend_label='10*EPS', color='orange', size=10, marker='triangle')
    
    p.toolbar.logo = None
    p.axis.minor_tick_in = -3
    p.legend.location = "top_left"
    p.legend.background_fill_alpha = 0.2
    p.legend.click_policy = "hide"
    return p


def process_explanation():
    st.markdown(
    """
    # Process strictly based on Graham's book The Intelligent Investor

    Our approach is built to be both simple and powerful—helping you quickly uncover undervalued stocks for further analysis.

    ## 1. Choose Your Approach  
    - **Search:** If you have a specific company in mind, simply enter its ticker to view a detailed value graph that overlays historical price trends with key financial metrics.  
    - **Screener:** If you’re exploring broadly, use the screener to sift through stocks that offer a significant margin of safety (think “net‑nets”) based on their tangible asset values.

    ## 2. Analyze the Fundamentals  
    - We check if a company’s assets are growing and if it has a history of profitability.  
    - We assess debt levels to ensure financial stability.  
    - Our app visually contrasts the current stock price with key value proxies (such as net current assets, earnings per share, and the Graham number) so you can spot valuation gaps at a glance.

    ## 3. Build Your Investment Case  
    - Once potential opportunities are flagged, dive into the company’s financial reports for deeper insights.  
    - Our built-in Graham classification system further evaluates stocks as Defensive, Enterprising, or Net‑Net, providing you with a structured framework to support your decision.

    ## 4. Explore and Decide  
    - Use these insights to refine your strategy and decide whether to investigate further or explore additional opportunities.

    **Give it a try and start discovering undervalued stocks at record speed!**
    """,
    unsafe_allow_html=True)


def salespage():
    if not st.session_state.get('license_valid', False):
        st.link_button("Get License Key", 'https://vysse.gumroad.com/l/ZeUmF')
        st.header("For Deep Value")
        st.markdown("""
        - **Cheap or Not:** Spot undervalued stocks quickly. 📉
        - **Fast Analysis:** Use our graphs for rapid insights. 🚀
        - **Save Time:** Quickly explore opportunities before deep dive. ⏳
        """)
        st.header("Avoid Investment Pitfalls")
        st.markdown("""
        - **Be Independent:** Stop depending on others. 🤝
        - **Stay Informed:** Understand the value trend. 📈
        - **Seize Opportunities:** Don’t miss out on great opportunities. 🚪
        """)
        st.header("Improve your process")
        st.markdown("""
        - **A learning opportunity:** Learn with graphs. 🏆
        - **Visualize Safety:** See the evolution of the margin of safety. 👀
        """)
        st.header("Accelerate your process")
        st.markdown("""
        - **Instant Analysis:** Generate price vs. value graphs quickly. ⚡
        - **Simple Screening:** Find and focus on potential bargains quickly. 🔍
        - **Work Smarter:** Explore 100 times faster to become autonomous. 🧠
        """)
        st.header("For You and Your Community")
        st.markdown("""
        - **Easy Explanations:** Save your graphs.  📊
        - **Share Ideas:** Discuss cases with clear graphs. 🗣️
        - **Independence:** Gain confidence and control. 🔒
        """)
        st.header("Why Now?")
        st.markdown("""
        - **Understand Your Process:** Learn from past successes and failures. 📚
        - **Deep Value:** Guide your actions with proven principles. 🧭
        - **Achieve Quality:** Hit your targets with minimal variation. 🎯
        """)
        st.header("Start now")
        st.markdown("""
        **Get your license key today** and unlock the full potential of ValeurGraph. 🗝️
        """)
        st.link_button("Get License Key", 'https://vysse.gumroad.com/l/ZeUmF')
    else:
        st.header("Thank you")
        st.markdown("""
        **You are an active user** thank you and enjoy ValeurGraph. 🗝️
        """)

# =============================================================================
# NEW: Sequential Stock Classification (Defensive > Enterprising > Net‑Net)
# =============================================================================
def get_first_value(df: pd.DataFrame, col: str, default_value: Any = np.nan) -> Any:
    if col in df.columns:
        series = df[col].dropna()
        if not series.empty:
            return series.iloc[0]
    return default_value

# Helper to append an evaluation criterion to the results list.
def check_and_append(results: List[List[Any]], criterion_name: str, condition: bool, score: Any) -> None:
    results.append([criterion_name, condition, score])

# Helper to display a value (return the value if not NaN, otherwise 'N/A').
def display_value(val: Any) -> Any:
    return val if not pd.isna(val) else 'N/A'

# Helper to compute potential as a percentage difference between a Graham number and the current price.
def compute_potential(graham: Any, current_price: Any) -> Any:
    try:
        if not pd.isna(graham) and not pd.isna(current_price) and current_price > 0:
            return round((graham - current_price) * 100 / current_price, 0)
    except Exception as e:
        logger.error(f"Error computing potential: {e}")
    return np.nan

def evaluate_defensive(data: pd.DataFrame, price: pd.DataFrame, dividends: pd.DataFrame) -> Tuple[pd.DataFrame, bool, Any]:
    """
    Evaluate whether a stock is defensive.
    Returns a 5-tuple: (LaTeX table, total score, evaluation details DataFrame, is_defensive flag, potential)
    """
    # --- Constants for Defensive Evaluation ---
    ANNUAL_SALES_THRESHOLD = 100_000_000
    CURRENT_ASSETS_MULTIPLIER = 2
    PRICE_EPS_MULTIPLIER = 15
    PRICE_BV_MULTIPLIER = 1.5
    RULE_OF_THUMB_THRESHOLD = 22.5
    EPS_GROWTH_THRESHOLD = 0
    REQUIRED_EPS_YEARS = 10
    REQUIRED_DIVIDEND_YEARS = 20
    AVG_EPS_YEARS = 3
    EPS_GROWTH_YEARS_START = 7

    results_def = []
    columns = data.columns

    # Retrieve key values
    graham_number = get_first_value(data, 'Graham_Number')
    annual_sales = get_first_value(data, 'AnnualSales', default_value=None)

    # Criterion 1: Annual Sales
    if annual_sales is not None:
        condition = annual_sales >= ANNUAL_SALES_THRESHOLD
        check_and_append(results_def, "Annual Sales >= $100M", condition, annual_sales)
        logger.debug(f"Annual Sales: {annual_sales} >= {ANNUAL_SALES_THRESHOLD}: {condition}")
    else:
        check_and_append(results_def, "Annual Sales >= $100M", False, np.nan)
        logger.warning("Annual Sales missing.")

    # Criterion 2A: Current Assets >= 2 * Current Liabilities
    if 'totalCurrentAssets' in columns and 'totalCurrentLiabilities' in columns:
        current_assets = get_first_value(data, 'totalCurrentAssets')
        current_liabilities = get_first_value(data, 'totalCurrentLiabilities')
        if current_assets is not None and current_liabilities is not None and current_liabilities != 0:
            condition = current_assets >= CURRENT_ASSETS_MULTIPLIER * current_liabilities
            ratio = round(current_assets / (CURRENT_ASSETS_MULTIPLIER * current_liabilities), 2)
            check_and_append(results_def, "Current Assets >= 2 * Current Liabilities", condition, ratio)
            logger.debug(f"Current Assets: {current_assets} >= 2*{current_liabilities}: {condition}")
        else:
            check_and_append(results_def, "Current Assets >= 2 * Current Liabilities", False, np.nan)
            logger.warning("Current Assets or Liabilities missing or zero.")
    else:
        check_and_append(results_def, "Current Assets >= 2 * Current Liabilities", False, np.nan)
        logger.warning("Current Assets or Current Liabilities columns missing.")

    # Criterion 2B: Net Current Assets >= Long-term Debt
    if all(col in columns for col in ['Net Current Asset/Non Current Liabilities', 'nonCurrentLiabilitiesTotal', 'totalCurrentAssets', 'totalLiab']):
        net_ratio = get_first_value(data, 'Net Current Asset/Non Current Liabilities')
        #net_current_assets = round(get_first_value(data, 'totalCurrentAssets') - get_first_value(data, 'totalLiab'), 2)
        #long_term_debt = get_first_value(data, 'nonCurrentLiabilitiesTotal')
        condition = net_ratio >= 1
        check_and_append(results_def, "Net Current Assets >= Non Current Liabilities", condition, round(net_ratio, 2))
        logger.debug(f"Net Ratio: {net_ratio} >= 1: {condition}")
    else:
        check_and_append(results_def, "Net Current Assets >= Non Current Liabilities", False, np.nan)
        logger.warning("Required columns for net current assets or lNon Current Liabilities missing.")

    # Criterion 3 & 5: EPS-related criteria
    if 'EPS' in columns:
        eps_series = data['EPS'].dropna().drop_duplicates().sort_index(ascending=False)
        if len(eps_series) >= REQUIRED_EPS_YEARS:
            eps_min_10yr = eps_series.iloc[:REQUIRED_EPS_YEARS].min()
            condition = eps_min_10yr > 0
            check_and_append(results_def, "Positive Earnings for 10 Years", condition, 'Yes' if condition else 'No')
            logger.debug(f"EPS 10yr min: {eps_min_10yr} > 0: {condition}")
            try:
                eps_start = eps_series.iloc[EPS_GROWTH_YEARS_START:REQUIRED_EPS_YEARS].mean()
                eps_end = eps_series.iloc[:AVG_EPS_YEARS].mean()
                growth = (eps_end - eps_start) / eps_start if eps_start > 0 else 0
                condition_growth = growth >= EPS_GROWTH_THRESHOLD
                growth_percentage = round(growth * 100, 2)
                check_and_append(results_def, "EPS Growth >= 1/3 in 10 Years", condition_growth, growth_percentage)
                logger.debug(f"EPS Growth: {growth_percentage}%: {condition_growth}")
            except Exception as e:
                check_and_append(results_def, "EPS Growth >= 1/3 in 10 Years", False, np.nan)
                logger.error(f"Error calculating EPS growth: {e}")
        else:
            check_and_append(results_def, "Positive Earnings for 10 Years", False, np.nan)
            check_and_append(results_def, "EPS Growth >= 1/3 in 10 Years", False, np.nan)
            logger.warning("Not enough EPS data for 10-year evaluation.")
    else:
        check_and_append(results_def, "Positive Earnings for 10 Years", False, np.nan)
        check_and_append(results_def, "EPS Growth >= 1/3 in 10 Years", False, np.nan)
        logger.warning("EPS column missing.")

    # Criterion 4: Uninterrupted dividends for 20 years
    if {'Year', 'Count'}.issubset(dividends.columns):
        try:
            current_year = datetime.datetime.now().year
            last_20_years = set(range(current_year - REQUIRED_DIVIDEND_YEARS, current_year))
            dividend_years = set(dividends.loc[dividends['Count'] > 0, 'Year'])
            missing_years = last_20_years - dividend_years
            condition = len(missing_years) == 0
            dividend_result = "Yes" if condition else f'No (Missing {len(missing_years)} years)'
            check_and_append(results_def, "Uninterrupted Dividends for 20 Years", condition, dividend_result)
            logger.debug(f"Dividends: missing {len(missing_years)}: {condition}")
        except Exception as e:
            check_and_append(results_def, "Uninterrupted Dividends for 20 Years", False, "No")
            logger.error(f"Error evaluating dividends: {e}")
    else:
        check_and_append(results_def, "Uninterrupted Dividends for 20 Years", False, "No")
        logger.warning("Dividends columns missing.")

    # Criterion 6: Current Price <= 15 * Average EPS (past 3 years)
    if 'EPS' in columns and not data['EPS'].dropna().empty:
        try:
            eps_clean = data['EPS'].dropna().drop_duplicates().sort_index(ascending=False)
            if len(eps_clean) >= AVG_EPS_YEARS:
                avg_eps_3yr = eps_clean.iloc[:AVG_EPS_YEARS].mean()
                current_price = price.iloc[-1]['adjusted_close']
                if avg_eps_3yr > 0:
                    price_eps_ratio = round(current_price / avg_eps_3yr, 2)
                    condition = price_eps_ratio <= PRICE_EPS_MULTIPLIER
                else:
                    price_eps_ratio = np.nan
                    condition = False
                check_and_append(results_def, "Price <= 15 * Avg EPS (3yr)", condition, price_eps_ratio)
                logger.debug(f"Price/EPS ratio: {price_eps_ratio} <= {PRICE_EPS_MULTIPLIER}: {condition}")
            else:
                check_and_append(results_def, "Price <= 15 * Avg EPS (3yr)", False, np.nan)
                logger.warning("Not enough EPS for 3-year average.")
        except Exception as e:
            check_and_append(results_def, "Price <= 15 * Avg EPS (3yr)", False, np.nan)
            logger.error(f"Error in Price/EPS criterion: {e}")
    else:
        check_and_append(results_def, "Price <= 15 * Avg EPS (3yr)", False, np.nan)
        logger.warning("EPS data missing for Price/EPS.")

    # Criterion 7: Current Price <= 1.5 * Book Value
    if 'BookValuePerShare' in columns:
        try:
            bvps = get_first_value(data, 'BookValuePerShare')
            current_price = price.iloc[-1]['adjusted_close']
            if bvps > 0:
                price_bv_ratio = round(current_price / bvps, 2)
                condition = price_bv_ratio <= PRICE_BV_MULTIPLIER
            else:
                price_bv_ratio = np.nan
                condition = False
            check_and_append(results_def, "Price <= 1.5 * Book Value", condition, price_bv_ratio)
            logger.debug(f"Price/BV ratio: {price_bv_ratio} <= {PRICE_BV_MULTIPLIER}: {condition}")
        except Exception as e:
            check_and_append(results_def, "Price <= 1.5 * Book Value", False, np.nan)
            logger.error(f"Error evaluating Book Value: {e}")
    else:
        check_and_append(results_def, "Price <= 1.5 * Book Value", False, np.nan)
        logger.warning("BookValuePerShare missing.")

    # Criterion 8: Multiplier * Price/Book <= 22.5
    if 'BookValuePerShare' in columns and 'EPS' in columns:
        try:
            # Retrieve results from earlier criteria (if available)
            price_eps_result = next((r for r in results_def if r[0] == "Price <= 15 * Avg EPS (3yr)"), None)
            #price_bv_result = next((r for r in results_def if r[0] == "Price <= 1.5 * Book Value"), None)
            multiplier = price_eps_result[2] if price_eps_result and not pd.isna(price_eps_result[2]) else np.nan
            bvps = get_first_value(data, 'BookValuePerShare')
            current_price = price.iloc[-1]['adjusted_close']
            price_bv_ratio = current_price / bvps if bvps > 0 else np.nan
            if not pd.isna(multiplier) and not pd.isna(price_bv_ratio):
                rule_of_thumb = multiplier * price_bv_ratio
                condition = rule_of_thumb <= RULE_OF_THUMB_THRESHOLD
            else:
                rule_of_thumb = np.nan
                condition = False
            check_and_append(results_def, "Multiplier * Price/Book <= 22.5", condition, round(rule_of_thumb, 2) if not pd.isna(rule_of_thumb) else np.nan)
            logger.debug(f"Rule of Thumb: {rule_of_thumb} <= {RULE_OF_THUMB_THRESHOLD}: {condition}")
        except Exception as e:
            check_and_append(results_def, "Multiplier * Price/Book <= 22.5", False, np.nan)
            logger.error(f"Error evaluating Price/Book multiplier: {e}")
    else:
        check_and_append(results_def, "Multiplier * Price/Book <= 22.5", False, np.nan)
        logger.warning("BookValuePerShare or EPS missing for Price/Book multiplier.")

    # Criterion 9: Graham Number Defensive (price <= Graham Number)
    if 'Graham_Number' in columns:
        try:
            graham_number_defensive = graham_number
            current_price = price.iloc[-1]['adjusted_close']
            condition = graham_number_defensive / current_price >= 1
            check_and_append(results_def, "Graham Number / Price >= 1", condition, round(graham_number_defensive / current_price, 2))
            logger.debug(f"Graham Number Defensive: {graham_number_defensive} / {current_price} >= 1: {condition}")
        except Exception as e:
            check_and_append(results_def, "Graham Number / Price >= 1", False, np.nan)
            logger.error(f"Error evaluating Graham Number Defensive: {e}")
    else:
        check_and_append(results_def, "Graham Number / Price >= 1", False, np.nan)
        logger.warning("Graham_Number missing.")

    # Build evaluation DataFrame and compute score.
    evaluation_df_def = pd.DataFrame(results_def, columns=['Criterion', 'Result', 'Score'])
    total_score = int(evaluation_df_def['Result'].sum())
    is_defensive = (total_score == 10)
    logger.info(f"Total Defensive Score: {total_score} out of 10")

    # Retrieve current price for potential calculation.
    try:
        current_price = price.iloc[-1]['adjusted_close']
    except Exception as e:
        current_price = np.nan
        logger.warning(f"Error retrieving current price: {e}")

    potential_def = compute_potential(graham_number, current_price)
    potential = f"{potential_def} %"

    return evaluation_df_def, is_defensive, potential

def evaluate_enterprising(data: pd.DataFrame, diluted_eps_ttm, price: pd.DataFrame, dividends: pd.DataFrame) -> Tuple[pd.DataFrame, bool, Any]:
    """
    Evaluate whether a stock is enterprising.
    Returns a 5-tuple: (LaTeX table, total score, evaluation DataFrame, is_enterprising flag, potential)
    """
    # --- Constants for Enterprising Evaluation ---
    CURRENT_ASSETS_MULTIPLIER_ENT = 1.5
    LONG_TERM_DEBT_MULTIPLIER_ENT = 1.1
    EPS_REQUIRED_YEARS_ENT = 5
    EPS_GROWTH_THRESHOLD_ENT = 0  # no minimal growth required
    PRICE_EPS_MULTIPLIER_ENT = 10
    NTAV_PRICE_MULTIPLIER_ENT = 1.2
    RULE_OF_THUMB_MULTIPLIER_ENT = 12

    results_ent = []
    columns = data.columns

    # Retrieve key values
    ntav = get_first_value(data, 'NTAV')
    graham_Number_Entp = get_first_value(data, 'Graham_Number_Entp')


    # Criterion 1-A: Current Assets >= 1.5 * Current Liabilities
    if 'totalCurrentAssets' in columns and 'totalCurrentLiabilities' in columns:
        try:
            current_assets = get_first_value(data, 'totalCurrentAssets')
            current_liabilities = get_first_value(data, 'totalCurrentLiabilities')
            condition = current_assets >= CURRENT_ASSETS_MULTIPLIER_ENT * current_liabilities
            ratio = round(current_assets / current_liabilities, 2) if current_liabilities != 0 else np.nan
            check_and_append(results_ent, "Current Assets >= 1.5 * Current Liabilities", condition, ratio)
            logger.debug(f"Enterprising Criterion 1-A: {current_assets} >= {CURRENT_ASSETS_MULTIPLIER_ENT} * {current_liabilities}: {condition}")
        except Exception as e:
            check_and_append(results_ent, "Current Assets >= 1.5 * Current Liabilities", False, np.nan)
            logger.error(f"Error evaluating Enterprising Criterion 1-A: {e}")
    else:
        check_and_append(results_ent, "Current Assets >= 1.5 * Current Liabilities", False, np.nan)
        logger.warning("Columns for Current Assets or Liabilities missing (Enterprising).")

    # Criterion 1-B: Net Current Asset/ Long Term Debt Ratio >= 1.1
    if 'Net Current Asset/Non Current Liabilities' in columns and 'nonCurrentLiabilitiesTotal' in columns:
        try:
            net_ratio = get_first_value(data, 'Net Current Asset/Non Current Liabilities')
            #net_current_assets = round(get_first_value(data, 'totalCurrentAssets') - get_first_value(data, 'totalLiab'), 2)
            #long_term_debt = get_first_value(data, 'nonCurrentLiabilitiesTotal')
            condition = net_ratio >= LONG_TERM_DEBT_MULTIPLIER_ENT
            check_and_append(results_ent, "Net Current Asset/Non Current Liabilities Ratio >= 1.1", condition, round(net_ratio, 2))
            logger.debug(f"Enterprising Criterion 1-B: Net Ratio {net_ratio} >= {LONG_TERM_DEBT_MULTIPLIER_ENT}: {condition}")
        except Exception as e:
            check_and_append(results_ent, "Net Current Asset/Non Current Liabilities Ratio >= 1.1", False, np.nan)
            logger.error(f"Error evaluating Enterprising Criterion 1-B: {e}")
    else:
        check_and_append(results_ent, "Net Current Asset/Non Current Liabilities Ratio >= 1.1", False, np.nan)
        logger.warning("Required columns missing for Enterprising Criterion 1-B.")

    # Criterion 2: Positive Earnings for the Past 5 Years
    if 'EPS' in columns:
        eps_series = data['EPS'].dropna()
        if len(eps_series) >= EPS_REQUIRED_YEARS_ENT:
            eps_5yr = eps_series.head(EPS_REQUIRED_YEARS_ENT).min()
            condition = eps_5yr > 0
            check_and_append(results_ent, "Positive Earnings for 5 Years", condition, 'Yes' if condition else 'No')
            logger.debug(f"Enterprising Criterion 2: EPS 5yr min {eps_5yr}: {condition}")
        else:
            check_and_append(results_ent, "Positive Earnings for 5 Years", False, np.nan)
            logger.warning("Not enough EPS for Enterprising Criterion 2.")
    else:
        check_and_append(results_ent, "Positive Earnings for 5 Years", False, np.nan)
        logger.warning("EPS column missing for Enterprising Criterion 2.")

    # Criterion 4: Dividends in the Last 12 Months
    if {'Year', 'Count'}.issubset(dividends.columns):
        try:
            one_year_ago = datetime.datetime.now() - datetime.timedelta(days=365)
            recent_dividends = dividends.loc[dividends['Year'] >= one_year_ago.year]
            condition_div = not recent_dividends.empty and (recent_dividends['Count'] > 0).any()
            dividend_result = "Yes" if condition_div else "No"
            check_and_append(results_ent, "Dividends in Last 12 Months", condition_div, dividend_result)
            logger.debug(f"Enterprising Criterion 4: Dividends in last 12 months: {condition_div}")
        except Exception as e:
            check_and_append(results_ent, "Dividends in Last 12 Months", False, "No")
            logger.error(f"Error evaluating Enterprising Criterion 4: {e}")
    else:
        check_and_append(results_ent, "Dividends in Last 12 Months", False, "No")
        logger.warning("Dividends columns missing for Enterprising Criterion 4.")

    # Criterion 4 (Alternate): EPS Growth from 4 Years Ago to Now
    current_year = datetime.datetime.now().year
    eps_4yr = np.nan
    eps_current = np.nan
    
    if 'EPS' in data.columns:
        eps_series = data['EPS'].dropna()
        
        if len(eps_series) >= 8:
            try:
                # Ensure the DataFrame index is datetime formatted
                data.index = pd.to_datetime(data.index)
    
                # Find EPS from 4 years ago
                eps_4yr_index = data.index[data.index.year == (current_year - 4)]
                
                if not eps_4yr_index.empty:
                    eps_4yr = data.loc[eps_4yr_index[0], 'EPS']  # Use first match
                    eps_current = diluted_eps_ttm
                    
                    # Compute growth
                    growth = (eps_current - eps_4yr) / eps_4yr if eps_4yr > 0 else 0
                    condition_growth = growth >= EPS_GROWTH_THRESHOLD_ENT
                    growth_percentage = round(growth * 100, 2)
                    
                    check_and_append(results_ent, "EPS Growth >= from 4 Years Ago", condition_growth, growth_percentage)
                    logger.debug(f"Enterprising Criterion 4: Growth from {eps_4yr} to {eps_current} = {growth_percentage}%: {condition_growth}")
                else:
                    check_and_append(results_ent, "EPS Growth >= from 4 Years Ago", False, np.nan)
                    logger.warning(f"No EPS data found for year {current_year - 4}.")
            
            except Exception as e:
                check_and_append(results_ent, "EPS Growth >= from 4 Years Ago", False, np.nan)
                logger.error(f"Error evaluating Enterprising EPS Growth: {e}")
        
        else:
            check_and_append(results_ent, "EPS Growth >= from 4 Years Ago", False, np.nan)
            logger.warning("Not enough EPS data for growth calculation (Enterprising).")
    
    else:
        check_and_append(results_ent, "EPS Growth >= from 4 Years Ago", False, np.nan)
        logger.warning("EPS column missing for growth calculation (Enterprising).")
    
    # Criterion 5: Current Price <= 10 * Average EPS (past 3 years)

    if diluted_eps_ttm > 0:
        try:
            current_price = price.iloc[-1]['adjusted_close']
            price_eps_ratio = round(current_price / diluted_eps_ttm, 2)
            condition = price_eps_ratio <= PRICE_EPS_MULTIPLIER_ENT
        except Exception as e:
            current_price = np.nan
            logger.warning(f"Error retrieving current price for Enterprising Criterion 5: {e}")
        
        check_and_append(results_ent, "Price <= 10 * Diluted EPS TTM", condition, price_eps_ratio)
        logger.debug(f"Enterprising Criterion 5: Price/EPS ratio: {price_eps_ratio} <= {PRICE_EPS_MULTIPLIER_ENT}: {condition}")
    else:
        check_and_append(results_ent, "Price <= 10 * Diluted EPS TTM", False, np.nan)
        logger.warning("Not enough data")

    # Criterion 6: Current Price <= 1.2 * NTAV
    if 'NTAV' in columns:
        try:
            ntav = get_first_value(data, 'NTAV')
            try:
                current_price = price.iloc[-1]['adjusted_close']
            except Exception as e:
                current_price = np.nan
                logger.warning(f"Error retrieving current price for Enterprising Criterion 6: {e}")
            if not pd.isna(ntav) and ntav > 0:
                ntav_ratio = round(current_price / ntav, 2)
                condition = ntav_ratio <= NTAV_PRICE_MULTIPLIER_ENT
                check_and_append(results_ent, "Price <= 1.2 * NTAV", condition, ntav_ratio)
                logger.debug(f"Enterprising Criterion 6: Price/NTAV: {ntav_ratio} <= {NTAV_PRICE_MULTIPLIER_ENT}: {condition}")
            else:
                check_and_append(results_ent, "Price <= 1.2 * NTAV", False, np.nan)
        except Exception as e:
            check_and_append(results_ent, "Price <= 1.2 * NTAV", False, np.nan)
            logger.error(f"Error evaluating Enterprising Criterion 6: {e}")
    else:
        check_and_append(results_ent, "Price <= 1.2 * NTAV", False, np.nan)
        logger.warning("NTAV column missing for Enterprising Criterion 6.")

    # Criterion 7: PER * NTAV <= 12
    if 'EPS' in columns and 'NTAV' in columns:
        eps_value = diluted_eps_ttm
        ntav = get_first_value(data, 'NTAV')
        if not pd.isna(eps_value) and not pd.isna(ntav):
            try:
                current_price = price.iloc[-1]['adjusted_close']
            except Exception as e:
                current_price = np.nan
                logger.warning(f"Error retrieving current price for Enterprising Criterion 7: {e}")
            multiplier = current_price / eps_value if eps_value > 0 else np.nan
            ntav_ratio = current_price / ntav if ntav > 0 else np.nan
            if not pd.isna(multiplier) and not pd.isna(ntav_ratio):
                rule_of_thumb = multiplier * ntav_ratio
                condition = rule_of_thumb <= RULE_OF_THUMB_MULTIPLIER_ENT
                check_and_append(results_ent, "PER * NTAV <= 12", condition, round(rule_of_thumb, 2))
                logger.debug(f"Enterprising Criterion 7: Rule of thumb: {rule_of_thumb} <= {RULE_OF_THUMB_MULTIPLIER_ENT}: {condition}")
            else:
                check_and_append(results_ent, "PER * NTAV <= 12", False, np.nan)
        else:
            check_and_append(results_ent, "PER * NTAV <= 12", False, np.nan)
            logger.warning("Not enough data for Enterprising Criterion 7.")
    else:
        check_and_append(results_ent, "PER * NTAV <= 12", False, np.nan)
        logger.warning("Required columns missing for Enterprising Criterion 7.")

    evaluation_df_ent = pd.DataFrame(results_ent, columns=['Criterion', 'Result', 'Score'])
    total_score = int(evaluation_df_ent['Result'].sum())
    is_ent = (total_score == 8)
    logger.info(f"Total Enterprising Score: {total_score} out of 8")

    try:
        current_price = price.iloc[-1]['adjusted_close']
    except Exception as e:
        current_price = np.nan
        logger.warning(f"Error retrieving current price (Enterprising): {e}")

    potential_ent = compute_potential(graham_Number_Entp, current_price)
    potential = f"{potential_ent} %"

    return evaluation_df_ent, is_ent, potential


def evaluate_netnet(data: pd.DataFrame, diluted_eps_ttm, price: pd.DataFrame)  -> Tuple[pd.DataFrame, bool, Any]:
    """
    Evaluate whether a stock is a Net-Net (undervalued based on Net Current Asset Value).
    Returns a 4-tuple: (LaTeX table, total score, evaluation details DataFrame, is_netnet flag)
    """
    results_netnet = []
    columns = data.columns

    # Retrieve key values
    ncav = get_first_value(data, 'NCAV')
    current_price = price.iloc[-1]['adjusted_close']
    #diluted_eps_ttm = get_first_value(data, 'netIncome')

    # Criterion 1: Price < NCAV
    if ncav is not None and current_price is not None:
        condition = current_price < ncav
        check_and_append(results_netnet, "Price < NCAV", condition, round(ncav / current_price, 2))
        logger.debug(f"Price {current_price} < NCAV {ncav}: {condition}")
    else:
        check_and_append(results_netnet, "Price < NCAV", False, np.nan)
        logger.warning("NCAV or Price missing.")

    # Criterion 2: Diluted EPS TTM is positive
    if diluted_eps_ttm is not None:
        condition = diluted_eps_ttm > 0
        check_and_append(results_netnet, "Diluted EPS TTM > 0", condition, diluted_eps_ttm)
        logger.debug(f"Diluted EPS TTM {diluted_eps_ttm} > 0: {condition}")
    else:
        check_and_append(results_netnet, "Diluted EPS TTM > 0", False, np.nan)
        logger.warning("Diluted EPS TTM missing.")

    # Build evaluation DataFrame and compute score
    evaluation_df_netnet = pd.DataFrame(results_netnet, columns=['Criterion', 'Result', 'Score'])
    total_score = int(evaluation_df_netnet['Result'].sum())
    is_net = (total_score == 2)
    potential_netnet = compute_potential(ncav, current_price)
    potential = f"{potential_netnet} %"
    logger.info(f"Total Net-Net Score: {total_score} out of 2")

    return evaluation_df_netnet, is_net, potential

def display_classification(ticker, financials, price, dividends, diluted_eps_ttm):
    """Classifies a stock and displays its potential price increase."""
    if not ticker:
        st.info("No stock selected for classification.")
        return
    
    with st.spinner("Evaluating stock classification..."):
        try:
            # Evaluate Defensive Classification
            def_summary, is_def, potential_def = evaluate_defensive(financials, price, dividends)
            if is_def:
                st.markdown("### Classification: it is Defensive")
                st.write(def_summary)
                st.markdown(f"**📈 Potential Price Increase:** {potential_def}")
                return
            
            st.markdown("### Why it is not Defensive:")
            st.write(def_summary)
            
            # Evaluate Enterprising Classification
            ent_summary, is_ent, potential_ent = evaluate_enterprising(financials, diluted_eps_ttm, price, dividends)
            if is_ent:
                st.markdown("### Classification: it is Enterprising")
                st.write(ent_summary)
                st.markdown(f"**📈 Potential Price Increase:** {potential_ent}")
                return
            
            st.markdown("### Why it is not Enterprising:")
            st.write(ent_summary)
            
            # Evaluate Net-Net Classification
            net_summary, is_net, potential_net = evaluate_netnet(financials, diluted_eps_ttm, price)
            if is_net:
                st.markdown("### Classification: it is a Net‑Net")
                st.write(round(net_summary, 2))
                st.markdown(f"**📈 Potential Price Increase:** {potential_net}")
            else:
                st.markdown("### Classification: Does not meet Defensive, Enterprising, or Net‑Net criteria")
                st.write(round(net_summary, 2))
        except Exception as e:
            st.error(f"Error during classification: {e}")

     
# =============================================================================
# Display graph and information
# =============================================================================
        

def display_graph():
    """Displays the value graph and stock classification for a given ticker."""
    ticker = st.session_state.get('selected_ticker', "")
    st.title(f'Value Graph:')
    
    query = st.text_input("Enter a stock ticker and click on Plot to see the value graph", ticker)
    ticker = query if not st.session_state.get('trigger_plot', False) else ticker

    if st.form_submit_button("Plot") or st.session_state.get('trigger_plot', False):
        st.session_state['trigger_plot'] = False
        try:
            with st.spinner('Loading data...'):
                # Fetch financials & stock price once, then reuse
                financials, country, dividends, diluted_eps_ttm = get_fundamentals(ticker)
                price = get_price_eod(ticker)

                if price.empty or financials.empty:
                    raise ValueError("No stock or fundamental data found")
                
                # Extract company name
                company_name = fetch_financials_with_country(ticker)[0].get('Name', ticker)
                
                # Generate and display graph
                bokeh_chart = create_bokeh_chart(company_name, financials, price)
                st.bokeh_chart(bokeh_chart, use_container_width=True)
                
                st.caption("ValeurGraph can make mistakes. Check important info.")
                
                if not st.session_state.get('license_valid', False):
                    st.markdown(':red[**To display the full value graph, get a license key**]')
                
                # Evaluate classification using pre-fetched data
                st.title("🔎 Graham Classification")
                display_classification(ticker, financials, price, dividends, diluted_eps_ttm)

                # Display full financial data
                st.dataframe(financials)
        except Exception as e:
            st.error(f"An error occurred: Invalid input. Ticker format is CODE.EXCHANGE. Details: {e}")
# =============================================================================
# Disclaimer
# =============================================================================
def disclaimer():
    st.markdown(
        """
        **User Disclaimer:**  
        Please note that all information provided by this application is intended solely for your informational and educational use and does not constitute personalized financial, investment, or legal advice. You are solely responsible for any decisions you make based on this information.

        **Investment Principles:**  
        The methods and criteria used by this app are based on widely recognized value investing concepts—including those popularized by *The Intelligent Investor*. These ideas are in the public domain and have been adapted for general analysis.

        **Third-Party Data & API Usage:**  
        The financial data and company information displayed are sourced from various third-party APIs and data providers. The accuracy, completeness, or timeliness of the data is not guaranteed
        
        **Your Responsibility:**  
        Use the insights provided by this app as a starting point for your own research. We strongly that you consult with professional financial advisors before making any investment or business decisions.

        """,
        unsafe_allow_html=True
    )

# =============================================================================
# Main Application
# =============================================================================
def main():
    hide_default_format = """
    <style>
    #MainMenu {visibility: hidden; }
    [data-testid="stToolbar"] {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
    st.markdown(hide_default_format, unsafe_allow_html=True)
    st.title('ValeurGraph. Simple.')
    st.header("""
              The app for Deep Value investors applying Benjamin Graham's principles👇
              """)
    st.markdown('**"The person that turns over the most rocks wins the game."**')
    st.markdown('''In just a few seconds, Grahamite value investors can compare the evolution of a stock price with fundamental data per share (book value, NCAV, etc.)
    and check if it qualifies as a deep value opportunity (Defensive, Enterprising or Net-Net). The screener further accelerates the process, while keeping things as simple as possible.''')
    custom_css = """
    <style>
    .stApp {
        background-image: linear-gradient(to top, #c8e0ff, #c4e7ff, #c3eeff, #c5f4fe, #caf9fb, #d0fbfc, #d7fdfe, #ddffff, #e7feff, #f2fdff, #fbfdff, #ffffff);
        background-size: cover;
    }
    [data-testid="stForm"] {
        background-color: white !important;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    h1 {
        font-family: 'Lobster', cursive;
        font-size: 36px;
        color: #2a7868;
        text-align: center;
        margin-top: 20px;
        text-shadow: 2px 2px #f9f871;
    }
    [data-testid="collapsedControl"] {
        background-color: white !important;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stApp .stForm > div {
        background-color: white !important;
    }
    </style>
    """
    #st.markdown(custom_css, unsafe_allow_html=True)
    with st.sidebar:
        st.header('Settings')
        st.link_button("Get License Key", 'https://vysse.gumroad.com/l/ZeUmF')
    if 'license_valid' not in st.session_state:
        license_key = st.sidebar.text_input("Enter your license key", type="password", autocomplete="license-key")
        if st.sidebar.button('Validate License'):
            valid, remaining_days, expiration_date = check_license(license_key)
            if valid:
                st.session_state['license_valid'] = True
                st.session_state['remaining_days'] = remaining_days
                if remaining_days is not None:
                    st.sidebar.success(f'✅ Your license will expire in {remaining_days} days on {expiration_date.strftime("%Y-%m-%d")}')
                else:
                    st.sidebar.success(f'✅ Your license is valid')
            else:
                st.session_state['license_valid'] = False
                if expiration_date:
                    st.sidebar.error(f'😢 Your license expired on {expiration_date.strftime("%Y-%m-%d")}, get a new one')
                else:
                    st.sidebar.error('Invalid License Key')
    twitter_button_html = """
    <a href="https://twitter.com/Vysse36?ref_src=twsrc%5Etfw" class="twitter-follow-button" data-show-count="false">Follow @Vysse36</a>
    <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
    """
    subscribe_html = '''<style>#gumroad-follow-form-embed{margin:0; padding:0; box-sizing:border-box; min-width:0; max-width:100%; vertical-align:bottom; background-clip:padding-box; display:grid; grid-auto-flow:column; row-gap:0.75rem; column-gap:0.75rem; grid-template-columns:1fr; grid-auto-columns:max-content; align-items:center;}#gumroad-follow-form-embed-button{margin:0; padding:0; box-sizing:border-box; min-width:0; max-width:100%; vertical-align:bottom; background-clip:padding-box; background:rgba(0,0,0,0); font-size:1rem; line-height:1.5; padding:0.75rem 1rem; border:solid 0.0625rem rgb(0 0 0/1); color:currentcolor; border-radius:0.25rem; display:inline-flex; align-items:center; justify-content:center; cursor:pointer; text-decoration:none; transition-duration:0.14s; transition-property:transform;background-color:rgb(0,0,0); color:rgb(255,255,255);}#gumroad-follow-form-embed-button:hover{transform:translate(-0.25rem,-0.25rem); box-shadow:0.25rem 0.25rem 0 rgb(0,0,0); background-color:rgb(255,144,232); color:rgb(0,0,0);}#gumroad-follow-form-embed-input{margin:0; padding:0; box-sizing:border-box; min-width:0; max-width:100%; vertical-align:bottom; background-clip:padding-box; padding:0.75rem 1rem; font-size:1rem; line-height:1.5; border:solid 0.0625rem rgb(0 0 0/1); border-radius:0.25rem; display:block; width:100%; background-color:rgb(255,255,255); color:rgb(0,0,0);}</style><form class="input-with-button" action="https://app.gumroad.com/follow_from_embed_form" method="post" id="gumroad-follow-form-embed"><input type="hidden" name="seller_id" value="3977192246300"/><input id="gumroad-follow-form-embed-input" type="email" placeholder="Your email address" name="email" value=""/><button class="primary" type="submit" id="gumroad-follow-form-embed-button">Follow</button></form>'''
    with st.sidebar:
        st.divider()
        st.markdown("""Screeners just give a snapshot. Value Graphs give trends in the blink of the eye.
                    Combined with our simplest Graham Screener you discover the best bargains at record speed.""")
        st.divider()
        st.header('Social media')
        components.html(twitter_button_html, height=50)
        components.html(subscribe_html)
        st.divider()
    with st.expander("⚙ Explanation of the process"):
        process_explanation()    
    with st.expander("🔎 Search Stock"):
        search_command()
    with st.expander("⏳ Screener"):
        with st.spinner("Loading data"):
            display_screener()      
    with st.form("Plot"):
        display_graph()
        #display_classification()  
    st.markdown("## Complete Monthly Report: Graham’s Criteria Worldwide")
    st.markdown(
    '''
    <a href="https://vysse.gumroad.com/l/DefensiveEnterprising?_gl=1*ggl3qx*_ga*MTk2OTk4NTAxNS4xNzM4NzgyNzA1*_ga_6LJN6D94N6*MTczOTYyODg0NC45LjEuMTczOTYyOTgyMS4wLjAuMA.." target="_blank">
        <button style="background-color: #007BFF; border: none; border-radius: 4px; color: white; padding: 10px 20px; font-size: 16px; cursor: pointer;">
            Get Complete Report
        </button>
    </a>
    ''',
    unsafe_allow_html=True)
    
    st.markdown(
        """
        Unlock this month's report featuring all companies that meet Graham’s criteria:
        - **Defensive**
        - **Enterprising**
        - **Net‑Net**
    
        This report is designed for intelligent and lazy investors who want to accelerate their process and quickly identify investment opportunities to investigate further.
        """
    )
    st.image("flipbook.gif", caption="Discover the opportunities", use_column_width=True)
    

    salespage()
    with st.expander("User Disclaimer", expanded=False):
        disclaimer()
    with st.expander("🌍 Net-net map"):
        filtered_data = load_data_NCAV()
        country_stock_count = stocks_per_country(filtered_data)
        st.title("Number of net-nets by Country")
        st.plotly_chart(netnetmap(country_stock_count), use_container_width=True, config={'displayModeBar': False})
        
if __name__ == "__main__":
    main()
