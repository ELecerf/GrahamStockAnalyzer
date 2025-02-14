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
