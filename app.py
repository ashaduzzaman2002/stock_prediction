from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st

# Streamlit page config
st.set_page_config(page_title='Stock Prediction',
                   page_icon='logo.png', initial_sidebar_state='auto')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title('Stock Trend Prediction')

start = '2010-01-01'
end = '2022-10-15'

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

try:
    # ✅ Replace pandas_datareader with yfinance
    df = yf.download(user_input, start=start, end=end)

    df = df.reset_index()
    df = df.drop(['Date'], axis=1)

    # Describing data
    st.subheader('Data from 2010 - 2022')
    st.write(df.describe())

    # Visualization
    st.subheader('Closing Price vs Time Chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time Chart with 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, 'r')
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, 'r')
    plt.plot(ma200, 'g')
    plt.plot(df.Close)
    st.pyplot(fig)

    # Splitting into training and testing data
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):len(df)])

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_arr = scaler.fit_transform(data_training)

    # Load ML model
    model = load_model('keras_model.h5')

    # Prepare testing data
    past_100_days = data_training.tail(100)
    final_df = past_100_days.append(data_testing, ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100: i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)

    scaler = scaler.scale_
    scaler_factor = 1 / scaler[0]
    y_predicted = y_predicted * scaler_factor
    y_test = y_test * scaler_factor

    # Prediction vs Original
    st.subheader('Prediction vs Original')
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_predicted, 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

except Exception as e:
    print(e)
    st.subheader('No data found')
