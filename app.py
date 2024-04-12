# app.py - This is your main Streamlit application file.

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.crypto_utils import fetch_all_coins_list, fetch_historical_data
from utils.image_classifier_utils import enhanced_preprocess_image, predict_digit, load_pre_train
from plotly.subplots import make_subplots
import plotly.graph_objs as go

from tensorflow.keras.models import load_model

from PIL import Image
import numpy as np

# Initialize Streamlit application
#st.title('Assignment Dashboard')

# Define tabs
tab1, tab2, tab3 = st.tabs(["Cryptocurrency Details", "Coin Comparison", "Image Classifier"])
coins_list = fetch_all_coins_list()
coin_names = {coin['id']: coin['name'] for coin in coins_list} if coins_list else {}

#Model Loading
model_path = 'digit_classifier_model.h5'
model = load_model(model_path)

# Code for Tab 1: Stock Details
with tab1:
    st.header("Cryptocurrency Details")

    # Attempt to fetch coins list
    try:
        #coins_list = fetch_all_coins_list()
        if not coins_list:
            st.error('Unable to fetch coins list. Please check your API or internet connection.')
            raise Exception('No data returned by the API.')
        
        coin_names = {coin['id']: coin['name'] for coin in coins_list}
        selected_coin_id = st.selectbox('Select a cryptocurrency:', options=[None] + list(coin_names.keys()), format_func=lambda x: coin_names.get(x, "Select a cryptocurrency..."))

        # If a coin is selected, fetch historical data
        if selected_coin_id:
            historical_data = fetch_historical_data(selected_coin_id)

            if historical_data:
                if historical_data and isinstance(historical_data, list) and len(historical_data) > 0:
                    df = pd.DataFrame(historical_data, columns=['timestamp', 'price'])
                    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Plotting the price over the last year using Plotly
                    fig = px.line(df, x='date', y='price', title=f'{coin_names[selected_coin_id]} Price Over the Last Year')
                    st.plotly_chart(fig)
                    
                    # Finding the max and min price along with their dates
                    max_price_row = df.loc[df['price'].idxmax()]
                    min_price_row = df.loc[df['price'].idxmin()]
                    
                    # Card structure for max and min prices
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                            <div style="padding:10px; border-radius:10px; border:1px solid #ccc; margin-bottom:10px;">
                                <h4 style="margin:0;">Highest Price:</h4>
                                <p style="margin:0;">${max_price_row['price']:.2f} on {max_price_row['date'].date()}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                            <div style="padding:10px; border-radius:10px; border:1px solid #ccc; margin-bottom:10px;">
                                <h4 style="margin:0;">Lowest Price:</h4>
                                <p style="margin:0;">${min_price_row['price']:.2f} on {min_price_row['date'].date()}</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    # If the historical_data is None or empty, display a message instead of breaking the application
                    st.error(f"Historical data for the selected cryptocurrency '{coin_names.get(selected_coin_id, selected_coin_id)}' is not available.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Code for Tab 2: Coin Comparison
with tab2:
    st.header("Cryptocurrency Coin Comparison")

    if not coins_list:
        st.error('Unable to fetch coins list. Please check your API or internet connection.')
        st.stop()
    
    col1, col2 = st.columns(2)
    with col1:
        selected_coin1_id = st.selectbox('Select the first cryptocurrency:', options=[None] + list(coin_names.keys()), format_func=lambda x: coin_names.get(x, "Select a cryptocurrency..."))
    with col2:
        selected_coin2_id = st.selectbox('Select the second cryptocurrency:', options=[None] + list(coin_names.keys()), format_func=lambda x: coin_names.get(x, "Select a cryptocurrency..."))

    time_frame = st.selectbox('Select time frame:', options=['1 week', '1 month', '1 year'])
    time_frame_mapping = {'1 week': 7, '1 month': 30, '1 year': 365}
    days = time_frame_mapping[time_frame]

    if st.button('Compare'):
        # Check if the same cryptocurrency has been selected
        if selected_coin1_id == selected_coin2_id:
            st.error("Please select two different cryptocurrencies for comparison.")
        else:
            # Fetch the historical data for the selected cryptocurrencies
            historical_data1 = fetch_historical_data(selected_coin1_id, days)
            historical_data2 = fetch_historical_data(selected_coin2_id, days)
            
            if historical_data1 and historical_data2:
                # Prepare the data for plotting
                df1 = pd.DataFrame(historical_data1, columns=['timestamp', 'price'])
                df1['date'] = pd.to_datetime(df1['timestamp'], unit='ms')
                
                df2 = pd.DataFrame(historical_data2, columns=['timestamp', 'price'])
                df2['date'] = pd.to_datetime(df2['timestamp'], unit='ms')

                # Create figure with secondary y-axis
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add the first cryptocurrency trace
                fig.add_trace(
                    go.Scatter(x=df1['date'], y=df1['price'], name=coin_names[selected_coin1_id]),
                    secondary_y=False,
                )

                # Add the second cryptocurrency trace
                fig.add_trace(
                    go.Scatter(x=df2['date'], y=df2['price'], name=coin_names[selected_coin2_id]),
                    secondary_y=True,
                )

                # Set titles and axis labels
                fig.update_layout(
                    title_text=f"Price Comparison of {coin_names[selected_coin1_id]} vs {coin_names[selected_coin2_id]} Over {time_frame}"
                )
                fig.update_xaxes(title_text="Date")
                fig.update_yaxes(title_text=f"{coin_names[selected_coin1_id]} Price (USD)", secondary_y=False)
                fig.update_yaxes(title_text=f"{coin_names[selected_coin2_id]} Price (USD)", secondary_y=True, overlaying='y', side='right')

                # Render the plot
                st.plotly_chart(fig)
            else:
                st.error("Failed to fetch historical data for one or both of the selected cryptocurrencies. Please try again later.")

with tab3:
    st.header("Digit Classifier")
    st.write("Upload an image of a digit (0-9), and the model will predict which digit it is.")

    # File uploader allows user to add their own image
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
         # Preprocess the uploaded image
        processed_image = enhanced_preprocess_image(uploaded_file)
        
        # Predict the digit from the image
        predicted_digit, confidence = predict_digit(load_pre_train(), processed_image)

        # Display the prediction and confidence level
        result_html = f"""
        <div style="color: green; font-size: 20px;">
            The model predicts this digit is a: <span style="color: red; font-weight: bold;">{predicted_digit}</span> 
            with confidence: <span style="color: red;">{confidence:.2f}</span>
        </div>
        """
        st.markdown(result_html, unsafe_allow_html=True)
        st.image(image, caption='Uploaded Image', use_column_width=True)

       