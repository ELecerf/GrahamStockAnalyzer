import streamlit as st
import pandas as pd
import requests
from pymongo import MongoClient
import datetime
#from datetime import datetime, timedelta
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
import os
import geopandas as gpd
import folium
import plotly
import country_converter as coco
import plotly.express as px


from PIL import Image
import streamlit.components.v1 as components

st.set_page_config(page_title="ValeurGraph", page_icon="📈")

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
@st.cache_data
def load_data_NCAV():
    with st.spinner('Loading data'):
        # Create a query filter to select documents where NCAV_0toMarketCap > 100
        query = {'NCAV_0toMarketCap': {'$gt': 100}}
        
        # Use the find method with the query
        cursor = Collection.find(query)
        df = pd.DataFrame(list(cursor))
        
        # Define the necessary columns
        columns = ['Name', 'Exchange', 'Code', 'close', 'GrahamNumberToPrice', 'NCAV_0toMarketCap', 'Country']
        df = df[columns]
        
        return df
@st.cache_data
def stocks_per_country(df):
    # Group by 'Country' and count the number of stocks
    country_stock_count = df.groupby('Country').size().reset_index(name='Net-Nets')
    cc = coco.CountryConverter()
    some_names = country_stock_count["Country"]
    standard_names = cc.convert(names = some_names, to = 'ISO3')
    country_stock_count["CountryISO"]=cc.convert(names = some_names, to = 'ISO3')
    return country_stock_count

def netnetmap(df):
    """
    Generate a Plotly figure for the given data.

    Args:
    data (dict): Dictionary containing 'Country', 'StockCount', 'CountryISO', 'Latitude', and 'Longitude' lists.

    Returns:
    plotly.graph_objs._figure.Figure: The generated Plotly figure.
    """

    # Create the choropleth map
    fig = px.choropleth(
        df,
        locations="CountryISO",
        color="Net-Nets",
        hover_name="Country",  # Use country names for hover
        hover_data={"Net-Nets": True, "CountryISO": False},  # Show StockCount, hide ISO codes
        color_continuous_scale=px.colors.sequential.Plasma
    )

    # Update layout to remove Plotly logo and add title
    fig.update_layout(
        dragmode=False,
        geo=dict(
            fitbounds="locations",
            visible=True),
        coloraxis_colorbar=dict(title="Net-Nets")  # Title for the color bar
    )
    return fig

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
	df['Sales']=round((df['totalRevenue'])/df['commonStockSharesOutstanding'],2)
	df['BV%']=round(df['BookValue']/df['totalAssets']*100,1)
	df['Liab%']=round(df['totalLiab']/df['totalAssets']*100,1)
	df['Current Assets/2*Current Liab'] = round(100*df['totalCurrentAssets']/(2*df['totalCurrentLiabilities']),2)
	df['Current Assets']=df['totalCurrentAssets']
	df['Net Current Asset/Non Current Liabilities']=round(100*(df['totalCurrentAssets']-df['totalLiab'])/df['nonCurrentLiabilitiesTotal'],2)
	df['2*equity/debt']=round(2*100*df['totalAssets']/df['totalLiab'])
	return df


def get_earnings(tick):
	url = "https://eodhistoricaldata.com/api/fundamentals/%s"%tick
	params = {'api_token': EOD_API_KEY, 'filter': "Financials::Income_Statement::yearly",'fmt':'json'}
	r = requests.get(url, params=params)
	r=r.json()
	df = pd.DataFrame.from_dict(r,orient='index')[['netIncomeApplicableToCommonShares','totalRevenue']]
	df.index=pd.to_datetime(df.index)
	df.index.names=['date']
	return df[:14]


def get_bsy_data(tick):
    columnOfInterest = ['commonStockSharesOutstanding','totalAssets','totalLiab','totalCurrentAssets',
                        'netTangibleAssets','cash','totalStockholderEquity','totalCurrentLiabilities','nonCurrentLiabilitiesTotal',]
    url = "https://eodhistoricaldata.com/api/fundamentals/%s"%tick
    params = {'api_token': EOD_API_KEY, 'filter': "Financials::Balance_Sheet",'fmt':'json'}
    r = requests.get(url, params=params)
    json=r.json()
    dfy=pd.DataFrame.from_dict(json['yearly'],orient="index")[columnOfInterest]
    dfq=pd.DataFrame.from_dict(json['quarterly'],orient="index")[columnOfInterest]
    dfy.index=pd.to_datetime(dfy.index)
    dfq.index=pd.to_datetime(dfq.index)
    df = pd.concat([dfq[:3],dfy])
    df.index=pd.to_datetime(df.index)
    df=df.sort_index(ascending=False)
    df.index.names=['date']
    return df[:14]


def get_fundamentals(tick):
    bsh=get_bsy_data(tick)
    ist=get_earnings(tick)
    df=bsh.join(ist, how='outer').drop_duplicates()
    df.index=pd.to_datetime(df.index)
    #df=df.sort_index(ascending=True)
    #df=df.iloc[::-1]
    df=CalcValues(df.astype(float))
    if st.session_state.get('license_valid', False):
        proxy=['Graham_Number','NCAV','10EPS','Sales','NTAV','BookValuePerShare','Current Assets/2*Current Liab',
               'Current Assets','Net Current Asset/Non Current Liabilities','2*equity/debt']
    else:
        proxy=['BookValuePerShare']
    return df[proxy]


def get_price_eod(tick):
	end = datetime.datetime.now()
	start = end - datetime.timedelta(days=4000)
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
    exchanges = ['PA', 'TSE', 'TO', 'MI', 'MC', 'US', 'AS', 'BR','WAR',
               'OL','CO','ST']  
    
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

    # Ensure there is at least one row to check
    if not data.empty:
        # Current assets should be at least twice current liabilities.
        if data.iloc[0]['Current Assets/2*Current Liab'] >= 100:
            score += 1
            score_details['Current Assets > 2*Current Liab'] = round(data.iloc[0]['Current Assets/2*Current Liab']/100,2)

    # Check for earnings in the past ten years
    cleaned_eps = data['10EPS'].dropna()
    if not cleaned_eps.empty and cleaned_eps.min() >= 0:
        score += 1
        score_details['Positive Earnings for 10 Years'] = 'True'
    else:
        score_details['Positive Earnings for 10 Years'] = 'False'
            

    if not data.empty and not price.empty:
        # Current assets should be at least twice current liabilities.
        if data.iloc[0]['NCAV']/price.iloc[-1]['adjusted_close'] >= 1:
            score += 1
        score_details['Price < NCAV'] = round(data.iloc[0]['NCAV']/price.iloc[-1]['adjusted_close'],2)
        
        # Long-term debt should not exceed the net current assets.
        if data.iloc[0]['Net Current Asset/Non Current Liabilities'] >= 100:
            score += 1
        score_details['Net Current Asset > Non Current Liabilities'] = round(data.iloc[0]['Net Current Asset/Non Current Liabilities']/100,2)

        if not cleaned_eps.empty and price.iloc[-1]['adjusted_close']/(cleaned_eps.iloc[0]/10) <= 15:
            score += 1
        score_details['Price < 15 EPS'] = round(price.iloc[-1]['adjusted_close']/(cleaned_eps.iloc[0]/10),2)

        if price.iloc[-1]['adjusted_close']/data.iloc[0]['BookValuePerShare'] <= 1.5:
            score += 1
        score_details['Price < 1.5 Book Value Per Share'] = round(price.iloc[-1]['adjusted_close']/data.iloc[0]['BookValuePerShare'],2)

    # Print detailed score using Streamlit
    st.markdown("## Graham scoring")
    for criterion, criterion_score in score_details.items():
        st.markdown(f"**{criterion}:** {criterion_score}")
    
    st.markdown(f"**Total Score: {score}/6**")

    return score


def create_bokeh_chart(stock,df_fundamentals, df_stock):
    # Prepare data sources
    s1 = ColumnDataSource(df_fundamentals)
    s2 = ColumnDataSource(df_stock)

    # Create a new plot with a title and axis labels
    p = figure(title = stock, x_axis_type='datetime', height=400, sizing_mode='stretch_width')

    # Add glyphs
    p.line('date', 'adjusted_close', source=s2, legend_label='Close price', color='midnightblue', alpha=0.8)
    p.scatter('date', 'BookValuePerShare', source=s1, legend_label='Book Value per share', color='red',marker='circle')
    p.scatter('date', 'NCAV', source=s1, legend_label='Net Current Asset Value per share', color='blue', size=10, marker='y')
    p.scatter('date', 'Graham_Number', source=s1, legend_label='Graham Number', color='green', marker='circle')
    p.scatter('date', 'NTAV', source=s1, legend_label='Net Tangible Asset Value per share', color='black',marker='circle')
    p.scatter('date', 'Cash', source=s1, legend_label='Cash', color='black',  size=10,marker='x')
    p.scatter('date', '10EPS', source=s1, legend_label='10*EPS', color='orange', size=10,marker='triangle')
    #p.scatter('date', 'Sales', source=s1, legend_label='Sales', color='green', size=10,marker='triangle')

    p.toolbar.logo = None
    p.axis.minor_tick_in = -3
    p.legend.location = "top_left"
    p.legend.background_fill_alpha = 0.2
    p.legend.click_policy="hide"
    return p
    

def display_graph():
    ticker = st.session_state.get('selected_ticker', "")
    st.title(f'Value graph')
    query = st.text_input("Enter a stock ticker and click on Plot to see the value graph", ticker)
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
                st.caption("ValeurGraph can make mistakes. Check important info.")
                if not st.session_state.get('license_valid', False):
                    st.markdown(':red[**To display the full value graph, get a license key**]')
                evaluate_company(df_fundamentals, df_stock)
                st.dataframe(df_fundamentals)
        except Exception as e:
            st.error(f"An error occurred: your input is not valid. Ticker format is CODE.EXCHANGE")

def process_explanation():
    st.markdown("""
    - **A good process is simple:** quickly find cheap opportunities to analyze further.
    - **Option 1:** you want to analyze a specific company with a value graph. **Use Search.**
    - **Option 2:** You want to turn over rocks to find cheap companies. **Use the screener.**
    
    The process I use is simple, I look for companies with a big margin of safety compared to their tangible assets. 
    --> **Net-nets** for example. 
    \nI check if their assets are growing and if they are profitable.
    \nThen I check that they have low debts.
    \nThen I look at the trends of the price vs. main value proxies like net current assets or earnings per share
    to visually see where we stand compared to historical ratios.
    \nThen I read their financial reports and investigate further to make a case. 
    \nThen I discuss it with fellow investors. It has worked very well for me.
    \n**And you? Give it a try**""")

def salespage():
    if not st.session_state.get('license_valid', False):
        st.link_button("Get License Key",'https://vysse.gumroad.com/l/ZeUmF')
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
        **Get your license key (free or paid) today** and unlock the full potential of ValeurGraph. 🗝️
        """)
        st.link_button("Get License Key",'https://vysse.gumroad.com/l/ZeUmF')
    else:
        st.header("Thank you")
        st.markdown("""
        **You are an active user** thank you and enjoy ValeurGraph. 🗝️
        """)
    

def main():
    
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
    st.header('The simplest app for Deep Value investors 👇')
    st.markdown('**"The person that turns over the most rocks wins the game."**')
    st.markdown('''In just a few seconds, Grahamite value investors can compare the evolution of a stock price with fundamental data per share (book value, NCAV, etc.)
    and check if it qualifies as a deep value opportunity. The screener further accelerates the process, while keeping things as simple as possible.''')
    #st.divider()
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
        st.link_button("Get License Key",'https://vysse.gumroad.com/l/ZeUmF')
    
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
    subscribe_html = '''<style>#gumroad-follow-form-embed{margin:
        0px; padding: 0px; box-sizing: border-box; min-width: 0px; max-width:
        100%; vertical-align: bottom; background-clip: padding-box; display: grid; grid-auto-flow: column;
        row-gap: 0.75rem; column-gap: 0.75rem; grid-template-columns: 1fr; grid-auto-columns: max-content;
        align-items: center;}#gumroad-follow-form-embed-button{margin: 0px; padding: 0px;
        box-sizing: border-box; min-width: 0px; max-width: 100%; vertical-align: bottom; background-clip: padding-box;
        background: rgba(0, 0, 0, 0); font-size: 1rem; line-height: 1.5; padding: 0.75rem 1rem; border: solid 0.0625rem rgb(0 0 0/1);
        color: currentcolor; border-radius: 0.25rem; "Mabry Pro", Avenir, Montserrat, Corbel, "URW Gothic", source-sans-pro, sans-serif;
        display: inline-flex; align-items: center; justify-content: center;
        row-gap: 0.5rem; column-gap: 0.5rem; cursor: pointer; text-decoration-style: solid;
        text-decoration-color: currentcolor; text-decoration: none; transition-timing-function: ease-out;
        transition-duration: 0.14s; transition-property: transform;background-color: rgb(0 0 0); color: rgb(255 255 255); }
        #gumroad-follow-form-embed-button:hover{transform: translate(-0.25rem, -0.25rem); box-shadow: 0.25rem 0.25rem 0rem rgb(0 0 0);
        background-color: rgb(255 144 232); color: rgb(0 0 0); }#gumroad-follow-form-embed-input{margin: 0px; padding: 0px;
        box-sizing: border-box; min-width: 0px; max-width: 100%; vertical-align: bottom; background-clip: padding-box;
        "Mabry Pro", Avenir, Montserrat, Corbel, "URW Gothic", source-sans-pro, sans-serif; padding: 0.75rem 1rem;
        font-size: 1rem; line-height: 1.5; border: solid 0.0625rem rgb(0 0 0/1); border-radius: 0.25rem; display:
        block; width: 100%; background-color: rgb(255 255 255); color: rgb(0 0 0); }
        #gumroad-follow-form-embed-input:disabled{cursor: not-allowed; opacity: 0.3;}#gumroad-follow-form-embed-input::placeholder{color:
        rgb(0 0 0/0.5);}#gumroad-follow-form-embed-input:focus-within{outline: 0.125rem solid rgb(255 144 232);}
        #gumroad-follow-form-embed-input:read-only{background-color: #f4f4f0;}</style><form class="input-with-button"
        action="https://app.gumroad.com/follow_from_embed_form" method="post" id="gumroad-follow-form-embed">
        <input type="hidden" name="seller_id" value="3977192246300"/><input id="gumroad-follow-form-embed-input"
        type="email" placeholder="Your email address" name="email" value=""/><button class="primary" type="submit"
        id="gumroad-follow-form-embed-button">Follow</button></form>'''
    with st.sidebar:
        st.divider()
        st.markdown("""Screeners just give a snapshot. Value Graphs give trends in the blink of the eye.
                    Combined with our simplest Graham Screener you discover the best bargains at record speed.""")
        st.divider()
        st.header('Social media')
        components.html(twitter_button_html, height=50)
        components.html(subscribe_html)
        st.divider()
        gumcode = """<script src="https://gumroad.com/js/gumroad.js"></script>
        <a class="gumroad-button" href="https://vysse.gumroad.com/l/ZeUmF" data-gumroad-overlay-checkout="true">Get on</a>"""
        #components.html(gumcode, height=600)

    #if st.session_state.get('license_valid', False):
    with st.expander("⚙ Explanation of the process"):
        process_explanation()    
    with st.expander("🔎 Search Stock"):
        search_command()
    with st.expander("⏳ Screener"):
        with st.spinner("load data"):
            display_screener()
    with st.expander("🌍 Net-net map"):
        # Load and filter the data
        filtered_data = load_data_NCAV()

        # Get the number of stocks per country
        country_stock_count = stocks_per_country(filtered_data)

        # Create and display the map
        st.title("Number of net-nets by Country")
        st.plotly_chart(netnetmap(country_stock_count), use_container_width=True, config={
        'displayModeBar': False  # Hide the mode bar which contains the Plotly logo
        })      
    with st.form("Plot"):
        display_graph()
    salespage()

    #components.html(gumcode, height=700)
        
# Run the app
if __name__ == "__main__":
    main()
