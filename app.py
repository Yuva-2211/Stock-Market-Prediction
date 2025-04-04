import numpy as np
import pandas as pd
import yfinance as yf 
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
model = load_model("/Users/yuvashankarnarayana/Desktop/eswar/stock_model.h5")

st.header("Stock market predictor")

stock = st.text_input("Enter stock symbol" , "AAPL")
start = '2010-01-01'
end = '2021-12-31'

data = yf.download(stock,start,end)

data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)]) #80% of data is used for training
data_test = pd.DataFrame(data.Close[int(len(data)*0.80) : len(data)])#20% of data used for testing

pas_100_days = data_train.tail(100) #last 100 days of training data
scaler = MinMaxScaler(feature_range = (0,1))

data_test = pd.concat((pas_100_days,data_test),ignore_index=True)
pas_100_days = data_train.tail(100)
data_test_scale = scaler.fit_transform(data_test)


st.subheader("Price vs MA50")

ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(10,8))
plt.plot(ma_50_days,"r")
plt.plot(data.Close,"g")
plt.show()
st.pyplot(fig1)


st.subheader("Price vs MA50 vs MA100")
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days,"r")
plt.plot(ma_100_days,"b")
plt.plot(data.Close,"g")
plt.show()
st.pyplot(fig2)


st.subheader("Price vs MA100 vs MA200")
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days,"r")
plt.plot(ma_200_days,"b")
plt.plot(data.Close,"g")
plt.show()
st.pyplot(fig3)


x = []
y = []

for i in range(100, int(data_test_scale.shape[0])):  # Use shape[0] for the number of rows
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])



x , y = np.array(x) , np.array(y)


predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale

y = y * scale



st.subheader("Original Price vs Predicted Price")

fig4 = plt.figure(figsize=(8,6))
plt.plot(predict,"r",label = "Predicted  Price")
plt.plot(y,"g",label = "Original Price")

plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Stock Price Prediction (Original vs Predicted)")
plt.show()
st.pyplot(fig4)


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math


mse = mean_squared_error(y, predict)


st.subheader(" Model Evaluation Metrics")
st.write(f" Mean Squared Error (MSE): `{mse:.4f}`")
