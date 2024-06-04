def search_command():
    st.title("Stock Search Tool")

    # Form for user input and search
    with st.form("Search Form"):
        query = st.text_input("Enter a stock symbol or name to search:", "")
        search_button = st.form_submit_button("Search")

    if search_button and query:
        with st.spinner("Searching for stocks..."):
            result_df = search_stocks(query)
            if not result_df.empty:
                st.write("Search Results:")
                
                # Display the dataframe with selectable rows
                selected_rows = st.dataframe(
                    result_df,
                    use_container_width=False,
                    hide_index=False,
                    selection_mode='single-row',
                    key='dataframe'
                )

                # Store the selected rows in the session state for further processing
                st.session_state['result_df'] = result_df
                st.session_state['selected_rows'] = selected_rows
            else:
                st.info("No results found for your search.")
    elif search_button:
        st.info("Please enter a query to search for stocks.")
    
    # Check if there is selected data in session state
    if 'selected_rows' in st.session_state:
        with st.form("Plot Form"):
            plot_button = st.form_submit_button("Plot selection")
            if plot_button:
                process_selection()

def process_selection():
    result_df = st.session_state.get('result_df')
    selected_rows = st.session_state.get('selected_rows')
    if selected_rows and 'selected_rows' in selected_rows:
        if selected_rows['selected_rows']:  # Check if any row is actually selected
            selected_index = selected_rows['selected_rows'][0]
            selected_row = result_df.iloc[selected_index]
            ticker = f"{selected_row['Code']}.{selected_row['Exchange']}"
            st.session_state['selected_ticker'] = ticker
            st.session_state['trigger_plot'] = True
            st.write(f"Selected: {ticker}")
        else:
            st.write("No row selected")
    else:
        st.write("Selection data not available")
