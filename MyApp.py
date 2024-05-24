import streamlit as st
import pandas as pd
import requests
from pymongo import MongoClient
import datetime
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
import os



# API Keys
FINNHUB_API_KEY = 'your_finnhub_api_key'
EOD_API_KEY = st.secrets["EOD_API_KEY"]
MONGO_DB = st.secrets["MONGO_DB"]
GUMROAD_API_URL = st.secrets["GUMROAD_API_URL"]

# Establish a connection to MongoDB
client = MongoClient(MONGO_DB)
db = client.valeurgraphDB

Collection = db["data"]
# Load data function
@st.cache_data
def load_data(exchanges=['TSE']):
    with st.spinner('loading data'):
        # Check if a list of exchanges is provided
        if exchanges:
            # Create a query filter to select documents where the 'Exchange' field is in the provided list
            query = {'Exchange': {'$in': exchanges}}
        else:
            # If no exchanges provided, retrieve all documents
            query = {}
        
        # Use the find method with the query
        cursor = Collection.find(query)
        df = pd.DataFrame(list(cursor))
        columns = ['Name', 'Exchange', 'Code', 'close', 'GrahamNumberToPrice', 'NCAV_0toMarketCap']
        return df[columns]

# License check
def check_license(key):
    # Check if the key is the hardcoded special keyword
    if key == 'daubasses':
        return True

    else:
        # Otherwise, verify the key using the Gumroad API
      GUMROAD_API_URL = "https://api.gumroad.com/v2/licenses/verify"
      PRODUCT_ID = "noBcgvvPwQDKj5lH5qZzDw=="  # Replace with your actual product ID
      params = {
          "product_id": PRODUCT_ID,
          "license_key": key
      }
      response = requests.post(GUMROAD_API_URL, data=params)
      result = response.json()
      return result.get("success", False)

# Function to fetch financial data

def fetch_financials(ticker):
    url = f"https://eodhistoricaldata.com/api/fundamentals/{ticker}"
    params = {
        "api_token": EOD_API_KEY,
        "fmt": "json"
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data
def CalcValues(df):
	df['Cash']=round((df['cash'])/df['commonStockSharesOutstanding'],2)
	df['NCAV']=round((df['totalCurrentAssets']-df['totalLiab'])/df['commonStockSharesOutstanding'],2)
	df['NTAV']=round((df['netTangibleAssets'])/df['commonStockSharesOutstanding'],2)
	df['BookValue']=round(df['totalStockholderEquity'],2)
	df['BookValuePerShare']=round(df['BookValue']/df['commonStockSharesOutstanding'],2)
	df['EPS']=round(df['netIncomeApplicableToCommonShares']/df['commonStockSharesOutstanding'],2)
	df['10EPS']=round(df['netIncomeApplicableToCommonShares']*10/df['commonStockSharesOutstanding'],2)
	df['EPS3']=df['EPS'].rolling(1).mean()
	df['Graham_Number']=round((22.5*df['BookValuePerShare'].clip(0)*df['EPS3'].clip(0))**0.5,2)
	df['BV%']=round(df['BookValue']/df['totalAssets']*100,1)
	df['Liab%']=round(df['totalLiab']/df['totalAssets']*100,1)
	return df


def get_earnings(tick):
	url = "https://eodhistoricaldata.com/api/fundamentals/%s"%tick
	params = {'api_token': EOD_API_KEY, 'filter': "Financials::Income_Statement::yearly",'fmt':'json'}
	r = requests.get(url, params=params)
	r=r.json()
	df = pd.DataFrame.from_dict(r,orient='index')[['netIncomeApplicableToCommonShares']]
	df.index=pd.to_datetime(df.index)
	df.index.names=['date']
	return df[:13]


def get_bsy_data(tick):
	columnOfInterest = ['commonStockSharesOutstanding','totalAssets','totalLiab','totalCurrentAssets','netTangibleAssets','cash','totalStockholderEquity']
	url = "https://eodhistoricaldata.com/api/fundamentals/%s"%tick
	params = {'api_token': EOD_API_KEY, 'filter': "Financials::Balance_Sheet",'fmt':'json'}
	r = requests.get(url, params=params)
	json=r.json()
	dfy=pd.DataFrame.from_dict(json['yearly'],orient="index")[columnOfInterest]
	dfq=pd.DataFrame.from_dict(json['quarterly'],orient="index")[columnOfInterest]
	dfy.index=pd.to_datetime(dfy.index)
	dfq.index=pd.to_datetime(dfq.index)
	df=pd.concat([dfq[:3],dfy])
	df.index=pd.to_datetime(df.index)
	df=df.sort_index(ascending=False)
	df.index.names=['date']
	return df[:10]


def get_fundamentals(tick):
	bsh=get_bsy_data(tick)
	ist=get_earnings(tick)
	df=bsh.join(ist).drop_duplicates()
	df.index=pd.to_datetime(df.index)
	#df=df.sort_index(ascending=True)
	#df=df.iloc[::-1]
	df=CalcValues(df.astype(float))
	df = df[['Graham_Number','NCAV','10EPS','NTAV','BookValuePerShare']]
	return df


def get_price_eod(tick):
	end = datetime.date.today()
	start = end - datetime.timedelta(days=3653)
	url = "https://eodhistoricaldata.com/api/eod/%s"%tick
	params = {'api_token': EOD_API_KEY, 'from':start,'to':end,'fmt':'json'}
	#r = requests.get(url,params=params).json()
	#r=r.json()
	price = pd.DataFrame.from_dict(requests.get(url,params=params).json())
	price = price.set_index('date')
	price.index = pd.to_datetime(price.index)
	price = price[['adjusted_close']]
	return price

def search_stocks(query):
    """Search stocks using the EOD Historical Data API."""
    if query:
        url = f'https://eodhistoricaldata.com/api/search/{query}'
        params = {
            'api_token': EOD_API_KEY,
            'type': 'stock',
            'limit': '30'
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            js = response.json()
            query_df = pd.DataFrame(js)
            return query_df
        else:
            st.error('Failed to fetch data from EOD Historical Data API')
            return pd.DataFrame()
    else:
        return pd.DataFrame()  # Return an empty DataFrame if the query is empty

def search_command():
    st.title("Stock Search Tool")

    # User input for stock symbol
    query = st.text_input("Enter a stock symbol or name to search:", "")

    # Search button to trigger the search
    if st.form_submit_button("Search"):
        if query:
            with st.spinner("Searching for stocks..."):
                result_df = search_stocks(query)
                if not result_df.empty:
                    st.write("Search Results:")
                    st.dataframe(result_df)
                else:
                    st.info("No results found for your search.")
        else:
            st.info("Please enter a query to search for stocks.")
# Main application
"""
def display_screener():
    st.title('Screener')
    # Form for exchange selection and data loading
    with st.form("Exchange Selector"):
        # Dropdown to select an exchange
        exchanges = ['PA', 'TSE', 'MI', 'AS']
        selected_exchange = st.selectbox('Select an exchange:', exchanges)
        # Submit button for the form
        submitted = st.form_submit_button("Load Data")
        
        if submitted:
            df = load_data([selected_exchange])
            selected_row = st.dataframe(df, selection_mode = 'single_row', on_select = 'rerun')# Display the loaded data
"""
def display_screener():
    st.title('Screener')

    # List of exchanges
    exchanges = ['PA', 'TSE', 'MI', 'AS']
    
    # Form for selecting an exchange and loading data
    with st.form("Exchange Selector"):
        selected_exchange = st.selectbox('Select an exchange:', exchanges)
        load_data_button = st.form_submit_button("Load Data")

    # If the form is submitted, load the data
    if load_data_button:
        df = load_data([selected_exchange])
        st.session_state['df'] = df  # Store the dataframe in session state to maintain state across reruns

    # Check if dataframe exists in session state
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        # Form for row selection and displaying selected row details
        with st.form("Row Selector"):
            # Display the dataframe with selectable rows
            selected_rows = st.dataframe(
                df,
                hide_index=False,
                selection_mode='single-row',
                key='dataframe'
            )
            display_row_button = st.form_submit_button("Display Selected Row Details")
            
            # Check if any row is selected and display the details
            if display_row_button:
                if selected_rows and 'selected_rows' in selected_rows:
                    selected_index = selected_rows['selected_rows'][0]  # Get the index of the selected row
                    selected_row = df.iloc[selected_index]  # Retrieve the selected row data
                    st.write(f"Code: {selected_row['Code']}, Exchange: {selected_row['Exchange']}")
                else:
                    st.write("No row selected")

def create_bokeh_chart(stock,df_fundamentals, df_stock):
    # Prepare data sources
    s1 = ColumnDataSource(df_fundamentals)
    s2 = ColumnDataSource(df_stock)

    # Create a new plot with a title and axis labels
    p = figure(title = stock, x_axis_type='datetime', height=400, sizing_mode='stretch_width')

    # Add glyphs
    p.line('date', 'adjusted_close', source=s2, legend_label='Close price', color='midnightblue', alpha=0.8)
    p.circle('date', 'BookValuePerShare', source=s1, legend_label='Book Value per share', color='red', line_width=1)
    p.y('date', 'NCAV', source=s1, legend_label='Net Current Asset Value per share', color='blue', line_width=1, size=10)
    p.circle('date', 'Graham_Number', source=s1, legend_label='Graham Number', color='green', line_width=1)
    p.circle('date', 'NTAV', source=s1, legend_label='Net Tangible Asset Value per share', color='black', line_width=1)
    p.x('date', 'Cash', source=s1, legend_label='Cash', color='black', line_width=1, size=10)
    p.triangle('date', '10EPS', source=s1, legend_label='10*EPS', color='orange', line_width=1, size=10)

    p.toolbar.logo = None
    p.axis.minor_tick_in = -3
    p.legend.location = "top_left"
    p.legend.background_fill_alpha = 0.2
    p.legend.click_policy="hide"
    return p

def display_graph():
    st.title('Value vs. Price graph')
    query = st.text_input("Enter a stock ticker", "")
    if st.form_submit_button("Plot"):
      user_input = query
      with st.spinner('loading graph...'):
        data = fetch_financials(user_input)
        #Basic data display
        name = data.get('General', {}).get('Name')
        st.write("Company Name:", name)
        ex = data.get('General', {}).get('Exchange')
        st.write("Exchange:", ex)   # Plotting stock price
        df_stock = get_price_eod(user_input)
        df_fundamentals = get_fundamentals(user_input)
        #remove the duplicate indexes
        #df_fundamentals = df_fundamentals[~df_fundamentals.index.duplicated(keep='first')]
        #fill missing values with the last valid value for fundamentals
        #combined_df = pd.concat([df_fundamentals, df_stock], axis=1)
        #combined_df = combined_df.fillna(method='ffill')
        #st.line_chart(combined_df)
        bokeh_chart = create_bokeh_chart(name,df_fundamentals, df_stock)
        st.bokeh_chart(bokeh_chart, use_container_width=True)



def main():
    st.title('Graham Stock Analyzer')
    st.header('"The person that turns over the most rocks wins the game."')
    st.sidebar.header('Settings')
	
    st.sidebar.link_button('I get my License Key','https://vysse.gumroad.com/l/ZeUmF')
    # License key check
    if 'license_valid' not in st.session_state:
        license_key = st.sidebar.text_input("Enter your license key", type="password",autocomplete="license-key")
        if st.sidebar.button('Validate License'):
            if check_license(license_key):
                st.session_state['license_valid'] = True
                st.experimental_rerun()
            else:
                st.sidebar.error('Invalid License Key')
		    
    if st.session_state.get('license_valid', False):
        with st.form("Search"):
          search_command()
        with st.form("Plot"):
          display_graph()
        with st.spinner("load dataframe"):
          display_screener()

# Run the app
if __name__ == "__main__":
    main()
