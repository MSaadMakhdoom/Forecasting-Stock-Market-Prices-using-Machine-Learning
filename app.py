from flask import Flask, render_template
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__)

model = keras.models.load_model('PSX_Model.h5')

# define the route for displaying the table
@app.route('/')
def show_table():
    data = []
    with open('PSX.csv', 'r') as file:
        df = pd.read_csv('PSX.csv')

        yesterday_price = df['Close'].shift(1)
        today_price = df['Close']
        price_change = today_price - yesterday_price
        percent_change = round((price_change / yesterday_price) * 100 ,1)

        # add the percent_change column to the DataFrame
        df['Percentage_Change_Price'] = percent_change

        yesterday_volume = df['Volume'].shift(1)
        today_volume = df['Volume']
        volume_change = today_volume - yesterday_volume
        percent_change_Volume = round((volume_change / yesterday_volume) * 100 ,1)

        # add the percent_change column to the DataFrame
        df['Percentage_Volume'] = percent_change_Volume

        last_row = df.tail(1).iloc[0]
        last_row = last_row.to_dict()

        data = df.filter(['Close'])
        # Convert the dataframe to a numpy array
        training_data_len = int(np.ceil( len(data) * .95 ))
        dataset = data.values
        # Scale the all of the data to be values between 0 and 1
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)

        # Create the scaled test data set
        test_data = scaled_data[training_data_len - 60: , :]
        # Create the data sets x_test and y_test
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])
            
        # Convert the data to a numpy array
        x_test = np.array(x_test)
        # Reshape the data
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
        # Get the models predicted price values 
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        valid = data[training_data_len:]
        valid['Predictions'] = predictions

        today_forcasting = valid.tail(1).iloc[0]
        today_forcasting = today_forcasting.to_dict()

        data = [row.to_dict() for _, row in df.iterrows()]
        data = [{k: str(v) for k, v in row.items()} for row in data]

    return render_template('index.html',data=data,today=last_row,forcast=today_forcasting)

    # return render_template('base.html',data=data,today=last_row,forcast=today_forcasting)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)



