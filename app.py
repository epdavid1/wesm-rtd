import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pickle import dump, load
import tensorflow
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

X_scaler = load(open('X_scaler.pkl', 'rb'))
y_scaler = load(open('y_scaler.pkl', 'rb'))

model = keras.models.load_model('model_358_12.h5')

st.title('WESM Load-Weighted Average Price Forecasting Tool')
test = pd.read_csv('test.csv')
#st.write(test)
#fig = px.line(test['price'][0:12])
#st.plotly_chart(fig)
for i in range(0,12):
  test_arr = test[i:i+12].to_numpy()
  forecast = y_scaler.inverse_transform(model.predict(X_scaler.transform(test_arr).reshape(1,12,3)))
  test.at[i+12, "price"] = forecast
historical_price = test[['price']].copy()
historical_price.loc[12:24, 'price'] = np.nan
forecasted_price = test[['price']].copy()
forecasted_price.loc[0:11, 'price'] = np.nan
price = pd.concat([historical_price, forecasted_price], axis=1)
price.columns = ['price', 'forecasted price']
#st.write(price)
fig = px.line(price)
fig.update_layout(
    xaxis_title="Relative time in minutes",
    yaxis_title="LWAP in PhP/MW"
    )
st.plotly_chart(fig)
data = st.file_uploader("Upload CSV file containing supply, demand, and price values. The first 12 rows should contain historical values of supply, demand, and price. The succeeding rows should only contain projected values of supply and demand which will be used to forecast the price.")
if data is not None:
    test = pd.read_csv(data)
    #st.write(test)
    #fig = px.line(test['price'][0:12])
    #st.plotly_chart(fig)
    for i in range(0,12):
      test_arr = test[i:i+12].to_numpy()
      forecast = y_scaler.inverse_transform(model.predict(X_scaler.transform(test_arr).reshape(1,12,3)))
      test.at[i+12, "price"] = forecast
    historical_price = test[['price']].copy()
    historical_price.loc[12:24, 'price'] = np.nan
    forecasted_price = test[['price']].copy()
    forecasted_price.loc[0:11, 'price'] = np.nan
    price = pd.concat([historical_price, forecasted_price], axis=1)
    price.columns = ['price', 'forecasted price']
    #st.write(price)
    fig = px.line(price)
    fig.update_layout(
    xaxis_title="Relative time in minutes",
    yaxis_title="LWAP in PhP/MW"
    )
    st.plotly_chart(fig)
