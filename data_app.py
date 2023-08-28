
import streamlit as st
import yfinance as yf
import numpy as np
import tensorflow as tf
from annotated_text import annotated_text
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import timedelta
import plotly.graph_objects as go
from datetime import date

# Fetch stock data from the yfinance API


def getData(ticker, start):
    today = date.today()
    return yf.download(ticker, start, today)

# Prepare data for model consumption


def processData(data):

    data = data["Close"].values.reshape(-1, 1)

    # scale the data for use by the lstm model
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataScaled = scaler.fit_transform(data)

    # apply rolling window of 60 days
    X, y = [], []
    for i in range(60, len(dataScaled)):
        X.append(dataScaled[i-60:i, 0])
        y.append(dataScaled[i, 0])

    # reshape data for the lstm model
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler


# Contructs and return the LSTM model


def contructLSTM(input):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(
        units=50, return_sequences=True, input_shape=input))
    model.add(tf.keras.layers.LSTM(units=50))
    model.add(tf.keras.layers.Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Returns future predictions made by the LSTM model


def predictFuture(model, sequence, scaler, days=10):
    futurePredictions = []

    currentSequence = sequence.copy()
    for _ in range(days):
        prediction = model.predict(currentSequence.reshape(1, -1, 1))
        futurePredictions.append(scaler.inverse_transform(prediction)[0][0])

        currentSequence = np.roll(currentSequence, -1)
        currentSequence[-1] = prediction

    return futurePredictions

# Main logic for streamlit applicaiton


def main():

    st.title("Stock Price Prediction")

    TICKER_OPTIONS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']

    tickerSymbol = st.selectbox(
        "Select a stock ticker symbol:", TICKER_OPTIONS)

    startDate = st.date_input(
        "Select start date for historical data:", date(2020, 1, 1))

    N = st.slider(label="Days to Predict", min_value=5, max_value=30, step=5)

    button = st.button(label="Go")

    if button:
        loadingMessage = st.empty()
        loadingMessage.text('Loading...\nThis can take up to 2 minutes')

        data = getData(tickerSymbol, startDate)
        X, y, scaler = processData(data)
        model = contructLSTM((X.shape[1], 1))
        model.fit(X, y, epochs=50, batch_size=32, verbose=0)

        # Predicitons for historical data
        predictedHistorical = model.predict(X)
        predictedHistorical = scaler.inverse_transform(predictedHistorical)
        residuals = data["Close"].iloc[60:].values - \
            predictedHistorical.ravel()

        # Predictions for future data
        predictedFuture = [data.index[-1] +
                           timedelta(days=i) for i in range(1, N+1)]
        forecastedPrices = predictFuture(model, X[-1], scaler, days=N)

        currentPrice = data["Close"].iloc[-1]
        change = ((forecastedPrices[-1] - currentPrice) / currentPrice) * 100

        st.write("Recommendation:")

        # Recommendation logic based on forecasted future price
        if change > 5:
            recommendation = "Strong Buy"
            st.markdown(
                "<div style='background-color: #388E3C; padding: 10px; border-radius: 10px;'>Strong Buy</div>", unsafe_allow_html=True)
        elif change > 2:
            recommendation = "Buy"
            st.markdown(
                "<div style='background-color: #4CAF50; padding: 10px; border-radius: 10px;'>Buy</div>", unsafe_allow_html=True)
        elif change > -2:
            recommendation = "Sell"
            st.markdown(
                "<div style='background-color: #E57373; padding: 10px; border-radius: 10px;'>Sell</div>", unsafe_allow_html=True)
        else:
            recommendation = "Strong Sell"
            st.markdown(
                "<div style='background-color: #D32F2F; padding: 10px; border-radius: 10px'>Strong Sell</div>", unsafe_allow_html=True)

        # Current, predicted, and forecasted price graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index[60:], y=data["Close"].iloc[60:], mode='lines', name='Actual Stock Price'))
        fig.add_trace(go.Scatter(x=data.index[60:], y=predictedHistorical.ravel(
        ), mode='lines', name='Predicted Stock Price'))
        fig.add_trace(go.Scatter(x=predictedFuture, y=forecastedPrices,
                      mode='lines+markers', name='Forecasted Stock Price', line=dict(dash='dash')))
        fig.update_layout(title=tickerSymbol + ': Actual, Predicted, and Forecasted Stock Prices',
                          xaxis_title='Date', yaxis_title='Stock Price', xaxis_rangeslider_visible=True)

        st.plotly_chart(fig)

        # Average daily return histogram
        data['Returns'] = data['Close'].pct_change()
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=data['Returns'], nbinsx=100))
        fig2.update_layout(title='Histogram of Daily Returns',
                           xaxis_title='Return', yaxis_title='Frequency')

        st.plotly_chart(fig2)

        data['Returns'] = data['Close'].pct_change()

        clean_returns = data['Returns'].dropna()

        # Volatitlity calculation based on average daily return standard dev.
        volatility = clean_returns.std()

        if volatility < 0.025:
            riskAssessment = "This stock has low volatility and is considered low risk."
        elif volatility < 0.05:
            riskAssessment = "This stock has medium volatility and is considered to have a moderate risk."
        else:
            riskAssessment = "This stock has high volatility and is considered high risk."

        st.write(riskAssessment)

        # Predicted vs Actual Scatterplot
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=data["Close"].iloc[60:], y=predictedHistorical.ravel(
        ), mode='markers', name='Predicted vs Actual'))
        fig3.update_layout(title='Predicted vs Actual Stock Prices',
                           xaxis_title='Actual Stock Prices', yaxis_title='Predicted Stock Prices')
        st.plotly_chart(fig3)

        # Calculate mean square error metric
        rmse = np.sqrt(mean_squared_error(
            data["Close"].iloc[60:], predictedHistorical))
        if rmse <= 10:
            rmseAssessment = "pass"
            rmseColor = "#388E3C"
        else:
            rmseAssessment = "fail"
            rmseColor = "#D32F2F"

        # Calculate mean ablsoute error metric
        mae = mean_absolute_error(
            data["Close"].iloc[60:], predictedHistorical)
        if mae <= 8:
            maeAssessment = "pass"
            maeColor = "#388E3C"
        else:
            maeAssessment = "fail"
            maeColor = "#D32F2F"

        # Calculate mean percentage error metric
        mape = np.mean(np.abs(
            (data["Close"].iloc[60:] - predictedHistorical.ravel()) / data["Close"].iloc[60:])) * 100
        if mape <= 3:
            mapeAssessment = "pass"
            mapeColor = "#388E3C"
        else:
            mapeAssessment = "fail"
            mapeColor = "#D32F2F"

        # Calculate r-squared metric
        r2 = r2_score(data["Close"].iloc[60:], predictedHistorical)
        if r2 >= .90:
            r2Assessment = "pass"
            r2Color = "#388E3C"
        else:
            r2Assessment = "fail"
            r2Color = "#D32F2F"

        # Set accuracy metrics color and status
        annotated_text("Root Mean Squared Error (RMSE): ",
                       (f"{rmse:.2f}", f"{rmseAssessment}", f"{rmseColor}"))
        annotated_text("Mean Absolute Error (MAE): ",
                       (f"{mae:.2f}", f"{maeAssessment}", f"{maeColor}"))
        annotated_text("Mean Absolute Percentage Error (MAPE): ",
                       (f"{mape:.2f}%", f"{mapeAssessment}", f"{mapeColor}"))
        annotated_text("Coefficient of Determination (R-Squared): ",
                       (f"{r2:.2f}", f"{r2Assessment}", f"{r2Color}"))

        # Actuals vs Residuals scatterplot
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=data["Close"].iloc[60:], y=residuals, mode='markers'))
        fig4.update_layout(title='Actual vs. Residuals',
                           xaxis_title='Actual Values', yaxis_title='Residuals')
        st.plotly_chart(fig4)

        # Set the loading text to an empty string
        loadingMessage.text('')


if __name__ == "__main__":
    main()
