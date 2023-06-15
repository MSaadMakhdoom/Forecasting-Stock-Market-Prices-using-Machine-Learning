from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import mlflow
import mlflow.keras

mlflow_dir = os.path.join(os.path.expanduser("~"), "mlruns")
os.makedirs(mlflow_dir, exist_ok=True)
os.environ['MLFLOW_TRACKING_URI'] = 'file://' + mlflow_dir


# Define the experiment
def get_experiment_id(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)

# Define the experiment
EXPERIMENT_NAME = "stock_price_prediction_model"
EXPERIMENT_ID = get_experiment_id(EXPERIMENT_NAME)

warnings.filterwarnings("ignore")

# Define the path to the CSV file relative to the DAG file
csv_file_path = os.path.join(os.path.dirname(__file__), 'PSX.csv')
data_dir_path = os.path.join(os.path.dirname(__file__), 'preprocessed_data')

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 6, 14)
}

dag = DAG(
    'stock_price_prediction_mlflow',
    default_args=default_args,
    description='DAG for stock price prediction',
    schedule_interval=None  # Set the schedule_interval as per your requirement
)

def preprocess_data():
    try:
        # Read the stock market data from a CSV file
        df = pd.read_csv(csv_file_path)
        print("Read file ....")
        # Set the index of the DataFrame to the 'Date' column as datetime
        df.index = pd.to_datetime(df.Date)

        # Remove the 'Date' column from the DataFrame
        del df['Date']

        # Extract the 'Close' column as the input data
        data = df.filter(['Close'])
        dataset = data.values

        # Calculate the training data length (95% of the dataset)
        training_data_len = int(np.ceil(len(dataset) * 0.95))
        print("Length of training data:", training_data_len)

        # Scale the data between 0 and 1 using MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        # Prepare the training data
        train_data = scaled_data[0:int(training_data_len), :]
        x_train = []
        y_train = []

        # Create training sequences with a window size of 60
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
            if i <= 61:
                print(x_train)
                print(y_train)
                print()

        # Convert the training data to numpy arrays and reshape for LSTM input
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Create the directory if it doesn't exist
        os.makedirs(data_dir_path, exist_ok=True)

        # Save each array separately
        np.save(os.path.join(data_dir_path, 'x_train.npy'), x_train)
        np.save(os.path.join(data_dir_path, 'y_train.npy'), y_train)
        np.save(os.path.join(data_dir_path, 'scaled_data.npy'), scaled_data)
        np.save(os.path.join(data_dir_path, 'dataset.npy'), dataset)

    except Exception as e:
        print(f"Error occurred in preprocess_data: {str(e)}")
        raise

def train_model():
    try:
        # Load the preprocessed data from the files
        x_train = np.load(os.path.join(data_dir_path, 'x_train.npy'))
        y_train = np.load(os.path.join(data_dir_path, 'y_train.npy'))
        scaled_data = np.load(os.path.join(data_dir_path, 'scaled_data.npy'))
        dataset = np.load(os.path.join(data_dir_path, 'dataset.npy'))

        # Create the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        # Compile and train the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name="run_1") as run:
            # Retrieve run id
            RUN_ID = run.info.run_id

            # Track parameters
            mlflow.log_param("optimizer", 'adam')
            mlflow.log_param("loss_function", 'mean_squared_error')
            mlflow.log_param("batch_size", 1)
            mlflow.log_param("epochs", 10)

            # Train the model and log metrics
            history = model.fit(x_train, y_train, batch_size=1, epochs=10)
            mlflow.log_metric("loss", history.history['loss'][-1])

            # Track model
            mlflow.keras.log_model(model, "PSX_Model")

        # Save the trained model
        model.save(os.path.join(data_dir_path, 'PSX_Model.h5'))        
        

 

    except Exception as e:
        print(f"Error occurred in train_model: {str(e)}")
        raise


def test_model():
    try:
        # Load the preprocessed data from the files
        _, _, scaled_data, dataset = np.load(os.path.join(data_dir_path, 'x_train.npy')), \
                                     np.load(os.path.join(data_dir_path, 'y_train.npy')), \
                                     np.load(os.path.join(data_dir_path, 'scaled_data.npy')), \
                                     np.load(os.path.join(data_dir_path, 'dataset.npy'))

        # Load the trained model
        model = load_model(os.path.join(data_dir_path, 'PSX_Model.h5'))
        
        training_data_len = int(np.ceil(len(dataset) * 0.95))

        # Prepare the test data
        test_data = scaled_data[training_data_len - 60:, :]
        x_test = []

        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Make predictions
        predictions = model.predict(x_test)

        # Fit the scaler with the training data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(dataset[:training_data_len])

        # Inverse transform the predictions
        predictions = scaler.inverse_transform(predictions)

        # Calculate RMSE
        y_test = dataset[training_data_len:, :]
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        print("Root Mean Squared Error:", rmse)

        with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name="run_2") as run:
            # Retrieve run id
            RUN_ID = run.info.run_id

            # Log RMSE metric
            mlflow.log_metric("rmse", rmse)

    except Exception as e:
        print(f"Error occurred in test_model: {str(e)}")
        raise

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

test_task = PythonOperator(
    task_id='test_model',
    python_callable=test_model,
    dag=dag
)

preprocess_task >> train_task >> test_task
