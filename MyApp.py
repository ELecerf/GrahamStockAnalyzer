import streamlit as st
import pandas as pd
import requests
from pymongo import MongoClient
import datetime
#from datetime import datetime, timedelta
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
import os

from PIL import Image
import streamlit.components.v1 as components



# API Keys
FINNHUB_API_KEY = 'your_finnhub_api_key'
EOD_API_KEY = st.secrets["EOD_API_KEY"]
MONGO_DB = st.secrets["MONGO_DB"]
GUMROAD_API_URL = st.secrets["GUMROAD_API_URL"]
PRODUCT_ID = st.secrets["PRODUCT_ID"]

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
        return True, None, None

    # Verify the key using the Gumroad API
    #GUMROAD_API_URL = "https://api.gumroad.com/v2/licenses/verify"
    #PRODUCT_ID = "noBcgvvPwQDKj5lH5qZzDw=="  # Replace with your actual product ID
    params = {
        "product_id": PRODUCT_ID,
        "license_key": key
    }
    response = requests.post(GUMROAD_API_URL, data=params)
    result = response.json()

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
    if not st.session_state.get('license_valid', False):
        proxy=['Graham_Number','NCAV','10EPS','NTAV','BookValuePerShare']
    else:
        proxy=['10EPS','BookValuePerShare']
        df = df[proxy]
    return df


def get_price_eod(tick):
	end = datetime.datetime.now()
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
@st.cache_data
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
    # Form for user input and search
    with st.form("Search Form"):
        st.title("Search company")
        query = st.text_input("Enter a stock symbol or name to search:", "")
        search_button = st.form_submit_button("Search")

    # Initialize result_df in session state if not already present
    if 'result_df' not in st.session_state:
        st.session_state['result_df'] = pd.DataFrame()

    # If search button is pressed
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

        # Form for plotting the selected row
        with st.form("Plot Form"):
            # Display the dataframe with selectable rows
            selected_rows = st.dataframe(
                result_df,
                use_container_width=False,
                hide_index=False,
                selection_mode='single-row',
                on_select='rerun',
                key='dataframeSearch'
            )

            plot_button = st.form_submit_button("Plot selection")

            # Check if any row is selected and display the details
            if plot_button:
                if selected_rows:
                    if selected_rows.selection['rows']:  # Check if any row is actually selected
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
        


# Main application

def display_screener():

    # List of exchanges
    exchanges = ['PA', 'TSE', 'MI', 'AS']
    
    # Form for selecting an exchange and loading data
    with st.form("Exchange Selector"):
        st.title('Screener')
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
                use_container_width=False,
                hide_index=False,
                selection_mode='single-row',
                on_select='rerun',
                key='dataframeScreener'
            )
            plot_button = st.form_submit_button("Plot selection")
            
            # Check if any row is selected and display the details
            if plot_button:
                if selected_rows and 'rows' in selected_rows.selection:
                    if selected_rows.selection['rows']:  # Check if any row is actually selected
                        selected_index = selected_rows.selection['rows'][0]
                        selected_row = df.iloc[selected_index]
                        ticker = f"{selected_row['Code']}.{selected_row['Exchange']}"
                        st.session_state['selected_ticker'] = ticker
                        st.session_state['trigger_plot'] = True
                        st.write(f"Selected: {ticker}")
                        # Reset the selection
                        #st.session_state['df'].at[selected_index, 'selected'] = False
                        
                    else:
                        st.write("No row selected")
                else:
                    st.write("Selection data not available")
                    # Reset the selection
                    #st.session_state['df'].at[selected_index, 'selected'] = False

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
    ticker = st.session_state.get('selected_ticker', "")
    st.title(f'Value graph')
    query = st.text_input("Enter a stock ticker", ticker)
    user_input = query
    if st.session_state.get('trigger_plot', False):
        user_input = ticker
    if st.form_submit_button("Plot")or st.session_state.get('trigger_plot', False):
        st.session_state['trigger_plot'] = False
        try:
            with st.spinner('Loading graph...'):
                data = fetch_financials(user_input)
                
                if not data or 'General' not in data or not data['General'].get('Name'):
                    raise ValueError("Invalid ticker or data not found")

                # Basic data display
                name = data['General'].get('Name')
                st.write("Company Name:", name)
                ex = data['General'].get('Exchange')
                st.write("Exchange:", ex)

                # Plotting stock price
                df_stock = get_price_eod(user_input)
                df_fundamentals = get_fundamentals(user_input)

                if df_stock.empty or df_fundamentals.empty:
                    raise ValueError("No stock or fundamental data found")

                bokeh_chart = create_bokeh_chart(name, df_fundamentals, df_stock)
                st.bokeh_chart(bokeh_chart, use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred: your input is not valid. Ticker format is CODE.EXCHANGE")

def salespage():
    if not st.session_state.get('license_valid', False):
        st.link_button("Buy License Key",'https://vysse.gumroad.com/l/ZeUmF')
        st.header("For deep value investors")
        st.markdown("""
        - **Identify Cheap Stocks:** Spot undervalued stocks quickly. 📉
        - **Fast Analysis:** Use our graphs for rapid insights. 🚀
        - **Save Time:** Skip hours of screening and Excel. ⏳
        """)

        st.header("Avoid Investment Pitfalls")
        st.markdown("""
        - **Be Independent:** Stop depending on others for stock picks. 🤝
        - **Stay Informed:** Understand the value trend of a company 📈
        - **Seize Opportunities:** Don’t miss out on great investments. 🚪
        """)

        st.header("Improve your process")
        st.markdown("""
        - **Proven Approach:** Used by successful value investors. 🏆
        - **Visualize Safety:** See the evolution of margin of safety. 👀
        """)

        st.header("Accelerate your process")
        st.markdown("""
        - **Instant Analysis:** Generate price vs. value graphs in milliseconds. ⚡
        - **Automated Screening:** Find and focus on bargains quickly. 🔍
        - **Work Smarter:** Analyze 100 times faster to become autonomous. 🧠
        """)

        st.header("For You and Your Community")
        st.markdown("""
        - **Easy Explanations:** Show your choices visually.  📊
        - **Share Ideas:** Discuss investments with clear graphs. 🗣️
        - **Financial Independence:** Gain confidence and control. 🔒
        """)

        st.header("Why Now?")
        st.markdown("""
        - **Understand Your Investments:** Learn from past successes and failures. 📚
        - **Deep Value Investing:** Guide your actions with proven principles. 🧭
        - **Achieve Quality:** Hit your targets with minimal variation. 🎯
        """)

        st.header("Start now")
        st.markdown("""
        **Buy your license key today** and unlock the full potential of ValeurGraph. 🗝️
        """)
        st.link_button("Buy License Key",'https://vysse.gumroad.com/l/ZeUmF')
    else:
        st.header("Thank you")
        st.markdown("""
        **You are an active user** thank you and enjoy ValeurGraph. 🗝️
        """)
    

def main():
    st.set_page_config(page_title="ValeurGraph", page_icon="📈")
    hide_default_format = """
    <style>
    #MainMenu {visibility: hidden; }
    [data-testid="stToolbar"]{
    visibility: hidden;
    }
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
    st.markdown(hide_default_format, unsafe_allow_html=True)
    st.title('ValeurGraph. Simple.')
    st.header('The simplest Deep Value app')
    st.markdown('**"The person that turns over the most rocks wins the game."**')
    st.divider()
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
    .stApp {
    .stForm > div {
        background-color: white !important;
    }
    </style>
    """
    #st.markdown(custom_css, unsafe_allow_html=True)
    with st.sidebar:
        st.header('Settings')
        st.link_button("Buy License Key",'https://vysse.gumroad.com/l/ZeUmF')
    
    # License key check
    if 'license_valid' not in st.session_state:
        license_key = st.sidebar.text_input("Enter your license key", type="password", autocomplete="license-key")
        if st.sidebar.button('Validate License'):
            valid, remaining_days, expiration_date = check_license(license_key)
            if valid:
                st.session_state['license_valid'] = True
                st.session_state['remaining_days'] = remaining_days
                if remaining_days is not None:
                    #st.rerun()
                    st.sidebar.success(f'✅ Your license will expire in {remaining_days} days on {expiration_date.strftime("%Y-%m-%d")}')   
                else:
                    #st.rerun()
                    st.sidebar.success(f'✅ Your license is valid')
            else:
                st.session_state['license_valid'] = False
                if expiration_date:
                    #st.rerun()
                    st.sidebar.error(f'😢 Your license expired on {expiration_date.strftime("%Y-%m-%d")}, get a new one')
                else:
                    #st.rerun()
                    st.sidebar.error('Invalid License Key')
    twitter_button_html = """
    <a href="https://twitter.com/Vysse36?ref_src=twsrc%5Etfw" class="twitter-follow-button" data-show-count="false">Follow @Vysse36</a>
    <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
    """
    with st.sidebar:
        st.divider()
        st.markdown("""Screeners just give a snapshot. Value Graphs give trends in the blink of the eye.
                    Combined with our simplest Graham Screener you discover the best bargains at record speed.""")
        st.divider()
        st.header('Social media')
        components.html(twitter_button_html, height=50)
        st.divider()
        gumcode = """<script src="https://gumroad.com/js/gumroad.js"></script>
        <a class="gumroad-button" href="https://vysse.gumroad.com/l/ZeUmF" data-gumroad-overlay-checkout="true">Buy on</a>"""
        #components.html(gumcode, height=600)

    #if st.session_state.get('license_valid', False):
    with st.expander("Search Stock"):
        search_command()
    with st.expander("Screener"):
        with st.spinner("load dataframe"):
            display_screener()
    with st.form("Plot"):
        display_graph()
    salespage()
    
    #components.html(gumcode, height=700)
        
# Run the app
if __name__ == "__main__":
    main()
