import streamlit as st
import pandas as pd
import plotly.express as px
from pickle import dump, load
import tensorflow
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

X_scaler = load(open('X_scaler.pkl', 'rb'))
y_scaler = load(open('y_scaler.pkl', 'rb'))

model = keras.models.load_model('model_358_12.h5')

st.title('Electricity Market Price Forecasting')
data = st.file_uploader("Upload excel file containing 12 rows of sequential historical supply, demand, and price values")
if data is not None:
    test = pd.read_csv(data)
    st.write(test)
    fig = px.line(test['price'])
    st.plotly_chart(fig)
    for i in range(0,12):
      test_arr = test[i:i+12].to_numpy()
      forecast = y_scaler.inverse_transform(model.predict(X_scaler.transform(test_arr).reshape(1,12,3)))
      test.at[i+12, "price"] = forecast
    st.write(test)
    fig = px.line(test['price'])
    st.plotly_chart(fig)