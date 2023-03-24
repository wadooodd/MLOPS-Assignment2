import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
from flask import Flask, render_template, request

# Create a Flask app
app = Flask(__name__)

# Define the route for the home page
@app.route('/')
def home():
    # Get the stock data
    symbol = 'AAPL'
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY' \
        '&symbol={}&interval=1min&apikey=PSAXD5PQOVJNQK7E'.format(symbol)
    response = requests.get(url)
    data = response.json()['Time Series (1min)']
    df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    for date in data:
        open_price = float(data[date]['1. open'])
        high_price = float(data[date]['2. high'])
        low_price = float(data[date]['3. low'])
        close_price = float(data[date]['4. close'])
        volume = float(data[date]['5. volume'])
        df = df.append({'date': date, 'open': open_price, 'high': high_price,
                        'low': low_price, 'close': close_price, 'volume': volume},
                       ignore_index=True)
    # Get the latest close price
    latest_close = df.loc[0, 'close']
    # Load the trained model from a pickle file
    model_filename = 'model.pkl'
    loaded_model = pickle.load(open(model_filename, 'rb'))
    # Make a prediction using the loaded model
    X = df[['open', 'high', 'low', 'volume']].values
    y = df['close'].values
    prediction = loaded_model.predict(X)
    mse = mean_squared_error(y, prediction)
    # Format the prediction and MSE as strings
    prediction_str = '${:,.2f}'.format(prediction[0])
    mse_str = '{:,.2f}'.format(mse)
    # Render the home page template with the data
    return render_template('index.html', latest_close=latest_close,
                           prediction=prediction_str, mse=mse_str)

# Define the route for the prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Load the trained model from a pickle file
    model_filename = 'model.pkl'
    loaded_model = pickle.load(open(model_filename, 'rb'))
    if request.method == 'POST':
        # Get the user input
        open_price = float(request.form['open'])
        high_price = float(request.form['high'])
        low_price = float(request.form['low'])
        volume = float(request.form['volume'])
        # Make a prediction using the loaded model
        X = np.array([[open_price, high_price, low_price, volume]])
        prediction = loaded_model.predict(X)
        # Format the prediction as a string
        prediction_str = '${:,.2f}'.format(prediction[0])
        # Render the prediction page template with the data
        return render_template('predict.html', prediction=prediction_str)
    else:
        # Render the prediction page template
        return render_template('predict.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
