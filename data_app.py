
import streamlit as st
import yfinance as yf
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import plotly.graph_objects as go
from datetime import date


def fetch_data(ticker_symbol, start_date):
    current_date = date.today()
    return yf.download(ticker_symbol, start_date, current_date)


def preprocess_data(data):
    try:
        data = data["Close"].values.reshape(-1, 1)

        if data.size == 0:
            raise ValueError(
                "Data is empty. Cannot proceed with preprocessing.")

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)

        X, y = [], []
        for i in range(60, len(data_scaled)):
            X.append(data_scaled[i-60:i, 0])
            y.append(data_scaled[i, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        return X, y, scaler

    except ValueError as error:
        st.error(f"An error occurred: {error}")


def build_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(
        units=50, return_sequences=True, input_shape=input_shape))
    model.add(tf.keras.layers.LSTM(units=50))
    model.add(tf.keras.layers.Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def predict_future(model, last_known_sequence, scaler, days_to_predict=10):
    future_predictions = []

    current_sequence = last_known_sequence.copy()
    for _ in range(days_to_predict):
        prediction = model.predict(current_sequence.reshape(1, -1, 1))
        future_predictions.append(scaler.inverse_transform(prediction)[0][0])

        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = prediction

    return future_predictions


def main():

    st.title("Stock Price Prediction")

    TICKER_OPTIONS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']

    ticker_symbol = st.selectbox(
        "Select a stock ticker symbol:", TICKER_OPTIONS)

    start_date = st.date_input(
        "Select start date for historical data:", date(2020, 1, 1))

    N = st.slider(label="Days to Predict", min_value=5, max_value=30, step=5)

    go_button = st.button(label="Go")

    if go_button:
        loading_message = st.empty()
        loading_message.text('Loading... Please wait.')

        data = fetch_data(ticker_symbol, start_date)
        X, y, scaler = preprocess_data(data)
        model = build_model((X.shape[1], 1))
        model.fit(X, y, epochs=50, batch_size=32, verbose=0)

        predicted_stock_price = model.predict(X)
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
        residuals = data["Close"].iloc[60:].values - \
            predicted_stock_price.ravel()

        future_dates = [data.index[-1] +
                        timedelta(days=i) for i in range(1, N+1)]
        future_predictions = predict_future(
            model, X[-1], scaler, days_to_predict=N)

        price_today = data["Close"].iloc[-1]
        predicted_change = (
            (future_predictions[-1] - price_today) / price_today) * 100

        st.write("Recommendation:")

        if predicted_change > 5:
            recommendation = "Strong Buy"
            st.markdown(
                "<div style='background-color: #388E3C; padding: 10px; border-radius: 10px;'>Strong Buy</div>", unsafe_allow_html=True)
        elif predicted_change > 2:
            recommendation = "Buy"
            st.markdown(
                "<div style='background-color: #4CAF50; padding: 10px; border-radius: 10px;'>Buy</div>", unsafe_allow_html=True)
        elif predicted_change > -2:
            recommendation = "Sell"
            st.markdown(
                "<div style='background-color: #E57373; padding: 10px; border-radius: 10px;'>Sell</div>", unsafe_allow_html=True)
        else:
            recommendation = "Strong Sell"
            st.markdown(
                "<div style='background-color: #D32F2F; padding: 10px; border-radius: 10px'>Strong Sell</div>", unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index[60:], y=data["Close"].iloc[60:], mode='lines', name='Actual Stock Price'))
        fig.add_trace(go.Scatter(x=data.index[60:], y=predicted_stock_price.ravel(
        ), mode='lines', name='Predicted Stock Price'))
        fig.add_trace(go.Scatter(x=future_dates, y=future_predictions,
                      mode='lines+markers', name='Forecasted Stock Price', line=dict(dash='dash')))
        fig.update_layout(title=ticker_symbol + ': Actual, Predicted, and Forecasted Stock Prices',
                          xaxis_title='Date',
                          yaxis_title='Stock Price',
                          xaxis_rangeslider_visible=True)

        st.plotly_chart(fig)

        data['Returns'] = data['Close'].pct_change()
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=data['Returns'], nbinsx=100))
        fig2.update_layout(title='Histogram of Daily Returns',
                           xaxis_title='Return', yaxis_title='Frequency')
        st.plotly_chart(fig2)

        data['Returns'] = data['Close'].pct_change()

        clean_returns = data['Returns'].dropna()

        volatility = clean_returns.std()

        if volatility < 0.025:
            risk_assessment = "This stock has low volatility and is considered low risk."
        elif volatility < 0.05:
            risk_assessment = "This stock has medium volatility and is considered to have a moderate risk."
        else:
            risk_assessment = "This stock has high volatility and is considered high risk."

        st.write(risk_assessment)

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=data["Close"].iloc[60:], y=residuals, mode='markers'))
        fig4.update_layout(title='Actual vs. Residuals',
                           xaxis_title='Actual Values', yaxis_title='Residuals')
        st.plotly_chart(fig4)

        rmse = np.sqrt(mean_squared_error(
            data["Close"].iloc[60:], predicted_stock_price))

        correlation = np.corrcoef(data["Close"].iloc[60:], residuals)[0, 1]
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)

        st.write(correlation, mean_residual, std_residual)

        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

        loading_message.text('Finished loading!')


if __name__ == "__main__":
    main()


# streamlit run /Users/jakewallace/VScode/Python/streamlit_apps/data_app.py
