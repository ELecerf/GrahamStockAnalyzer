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
    
    # Combine Financial Data
    financials = balance_sheet_df.join(income_statement_df, how='outer').drop_duplicates()
    financials.index = pd.to_datetime(financials.index)
    financials = financials.sort_index(ascending=False)
    financials = financials.apply(pd.to_numeric, errors='coerce')
    financials = financials[:16]
    
    return financials, country, dividends

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
    df['Net Current Asset/ Long Term Debt'] = round((df['totalCurrentAssets'] - df['totalLiab']) / df['longTermDebt'], 2)
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
        financials, country, dividends = fetch_financials_with_country(tick)
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
            'Graham_Number', 'NCAV', 'Cash', '10EPS', 'AnnualSales', 'NTAV',
            'BookValuePerShare', 'Current Assets/2*Current Liab',
            'Current Assets', 'Net Current Asset/ Long Term Debt'
        ]
    else:
        selected_metrics = ['BookValuePerShare']
    
    return financials[selected_metrics]

def get_full_fundamentals(tick):
    """
    Retrieves and processes the complete financials DataFrame without license-based filtering.
    This is used for plotting so that all necessary metrics are available.
    """
    try:
        financials, country, dividends = fetch_financials_with_country(tick)
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


def display_graph():
    ticker = st.session_state.get('selected_ticker', "")
    st.title(f'Value graph {ticker}')
    query = st.text_input("Enter a stock ticker and click on Plot to see the value graph", ticker)
    user_input = query
    if st.session_state.get('trigger_plot', False):
        user_input = ticker
    # For graphing, use the full financials (ignoring license filtering)
    if st.form_submit_button("Plot") or st.session_state.get('trigger_plot', False):
        st.session_state['trigger_plot'] = False
        try:
            with st.spinner('Loading graph...'):
                # Use get_full_fundamentals so all needed metrics are available for plotting.
                df_fundamentals = get_full_fundamentals(user_input)
                df_stock = get_price_eod(user_input)
                if df_stock.empty or df_fundamentals.empty:
                    raise ValueError("No stock or fundamental data found")
                # For display purposes, fetch company info from the full financial data.
                # (Alternatively, you could extract the company name from another source.)
                financial_data, _, _ = fetch_financials_with_country(user_input)
                company_name = financial_data.get('Name', user_input)
                bokeh_chart = create_bokeh_chart(company_name, df_fundamentals, df_stock)
                st.bokeh_chart(bokeh_chart, use_container_width=True)
                st.caption("ValeurGraph can make mistakes. Check important info.")
                if not st.session_state.get('license_valid', False):
                    st.markdown(':red[**To display the full value graph, get a license key**]')
                evaluate_company(df_fundamentals, df_stock)
                st.dataframe(df_fundamentals)
        except Exception as e:
            st.error(f"An error occurred: your input is not valid. Ticker format is CODE.EXCHANGE. Details: {e}")

def process_explanation():
    st.markdown("""
    - **A good process is simple:** quickly find cheap opportunities to analyze further.
    - **Option 1:** you want to search for a specific company and check the value graph. **Use Search.**
    - **Option 2:** You want to turn over rocks to find potentially undervalued companies. **Use the screener.**
    
    You can look for companies with a big margin of safety compared to their tangible assets. 
    --> **Net-nets** for example. 
    \nI check if their assets are growing and if they are profitable.
    \nThen Check that they have low debts.
    \nThen Look at the trends of the price vs. main value proxies like net current assets or earnings per share
    to visually see where it stands compared to historical ratios.
    \nThen Read their financial reports and investigate further to make a case. 
    \nThen Continue to explore.
    \n**Give it a try**""")

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
def evaluate_defensive(financials, price):
    row = financials.iloc[0]
    score = 0
    if 'AnnualSales' in financials.columns and row['AnnualSales'] >= 100_000_000:
        score += 1
    if 'Current Assets/2*Current Liab' in financials.columns and row['Current Assets/2*Current Liab'] >= 100:
        score += 1
    if 'NCAV' in financials.columns and not price.empty:
        if row['NCAV'] / price.iloc[-1]['adjusted_close'] >= 1:
            score += 1
    if '10EPS' in financials.columns and not price.empty:
        if price.iloc[-1]['adjusted_close'] / (row['10EPS'] / 10) <= 15:
            score += 1
    if 'BookValuePerShare' in financials.columns and not price.empty:
        if price.iloc[-1]['adjusted_close'] / row['BookValuePerShare'] <= 1.5:
            score += 1
    if 'Net Current Asset/Non Current Liabilities' in financials.columns and row['Net Current Asset/Non Current Liabilities'] >= 100:
        score += 1
    summary = f"Defensive Score: {score}/6"
    is_defensive = (score == 6)
    return summary, score, is_defensive

def evaluate_enterprising(financials, price):
    row = financials.iloc[0]
    score = 0
    if 'Current Assets/2*Current Liab' in financials.columns and row['Current Assets/2*Current Liab'] >= 75:
        score += 1
    if '10EPS' in financials.columns and not price.empty:
        if price.iloc[-1]['adjusted_close'] / (row['10EPS'] / 10) <= 10:
            score += 1
    if 'NTAV' in financials.columns and not price.empty:
        if price.iloc[-1]['adjusted_close'] / row['NTAV'] <= 1.2:
            score += 1
    if 'BookValuePerShare' in financials.columns and not price.empty:
        if price.iloc[-1]['adjusted_close'] / row['BookValuePerShare'] <= 1.2:
            score += 1
    summary = f"Enterprising Score: {score}/4"
    is_enterprising = (score == 4)
    return summary, score, is_enterprising

def evaluate_netnet(financials, price):
    row = financials.iloc[0]
    score = 0
    if 'NCAV' in financials.columns and not price.empty:
        if row['NCAV'] > price.iloc[-1]['adjusted_close']:
            score += 1
    if 'DilutedEpsTTM' in financials.columns and row['DilutedEpsTTM'] > 0:
        score += 1
    summary = f"Net-Net Score: {score}/2"
    is_netnet = (score == 2)
    return summary, score, is_netnet

def display_classification():
    ticker = st.session_state.get('selected_ticker', "")
    if not ticker:
        st.info("No stock selected for classification.")
        return
    with st.spinner("Evaluating stock classification..."):
        try:
            financials = get_fundamentals(ticker)
            price = get_price_eod(ticker)
        except Exception as e:
            st.error(f"Error fetching data for classification: {e}")
            return
        def_summary, def_score, is_def = evaluate_defensive(financials, price)
        if is_def:
            st.markdown("### Classification: Defensive")
            st.write(def_summary)
        else:
            ent_summary, ent_score, is_ent = evaluate_enterprising(financials, price)
            if is_ent:
                st.markdown("### Classification: Enterprising")
                st.write(ent_summary)
            else:
                net_summary, net_score, is_net = evaluate_netnet(financials, price)
                if is_net:
                    st.markdown("### Classification: Net‑Net")
                    st.write(net_summary)
                else:
                    st.markdown("### Classification: Does not meet Defensive, Enterprising, or Net‑Net criteria")

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
    st.header('The simplest app for Deep Value investors 👇')
    st.markdown('**"The person that turns over the most rocks wins the game."**')
    st.markdown('''In just a few seconds, Grahamite value investors can compare the evolution of a stock price with fundamental data per share (book value, NCAV, etc.)
    and check if it qualifies as a deep value opportunity. The screener further accelerates the process, while keeping things as simple as possible.''')
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
        st.title("🔎 Graham Classification")
        display_classification()  
    salespage()
    with st.expander("🌍 Net-net map"):
        filtered_data = load_data_NCAV()
        country_stock_count = stocks_per_country(filtered_data)
        st.title("Number of net-nets by Country")
        st.plotly_chart(netnetmap(country_stock_count), use_container_width=True, config={'displayModeBar': False})
        
if __name__ == "__main__":
    main()
