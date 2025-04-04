
# 📈 Stock Market Predictor (LSTM + Streamlit)

This project is a **Stock Price Predictor** built using **LSTM (Long Short-Term Memory)** neural networks and wrapped in a beautiful, interactive **Streamlit** web app. It allows users to enter a stock ticker (like `AAPL`, `GOOGL`, etc.) and view both historical trends and predicted stock prices based on past data.

---

## 🚀 Features

- 📊 Visualizes Moving Averages (50, 100, 200 days)
- 🔍 Interactive input to search any stock ticker
- 🤖 Predicts future prices using LSTM deep learning model
- 📉 Compares actual vs predicted stock prices
- 📈 Displays evaluation metrics: MSE
- 🖥️ Streamlit app with interactive charts and results

---

## 🛠️ Tech Stack

- Python 🐍
- Streamlit 🌐
- LSTM via Keras (TensorFlow backend)
- YFinance (to fetch stock data)
- Scikit-learn (for scaling and evaluation metrics)
- Matplotlib (for plotting)

---

## 📂 Project Structure

```
├── stock_model.h5           # Trained LSTM model
├── app.py                   # Streamlit web app code
├── README.md                # Project documentation
```

---

## ⚙️ How It Works

1. **Data Collection**: Downloads stock price data from Yahoo Finance (2010–2021).
2. **Preprocessing**: Applies MinMax scaling and creates sequences of 100-day windows.
3. **Model Training**: Uses a 4-layer stacked LSTM model to learn patterns.
4. **Prediction**: Predicts future stock prices using test data.
5. **Visualization**: Displays actual vs predicted prices, moving averages, and model evaluation.

---

## 📷 Sample Output (in Streamlit)

- Price vs Moving Averages  
- Actual vs Predicted Price  
- Model Evaluation:
  - Mean Squared Error (MSE)
---

## 🚀 How to Run Locally

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/stock-predictor-app.git
   cd stock-predictor-app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## 📦 Requirements

Create a `requirements.txt` using:

```
streamlit
numpy
pandas
yfinance
keras
matplotlib
scikit-learn
```

---

## ✨ Future Enhancements

- Add multi-stock comparison
- Add feature to predict n-days into the future
- Deploy the app on Streamlit Cloud or Render
- Include more technical indicators (RSI, MACD, etc.)

---

## 🙌 Technical summary 

- LSTM Model: Powered by Keras + TensorFlow
- Financial Data: Yahoo Finance via `yfinance`
- App Interface: Built using Streamlit

---

## 📬 Contact

Built with ❤️ by D. Yuva Shankar Narayana & Pavithra.H  
[LinkedIn(Pavithra H )](#https://www.linkedin.com/in/pavithra-h-048a8b321/) | 
[LinkedIn(Yuva Shankar Narayana )](https://www.linkedin.com/in/yuva-shankar-narayana-16b09a314) |
