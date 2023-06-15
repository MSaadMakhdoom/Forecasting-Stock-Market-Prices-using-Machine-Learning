[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
This project utilizes  Python programing for building the application

[![Docker](https://img.shields.io/badge/Docker-Containerization-blue)](https://www.docker.com/)
This project utilizes Docker for containerizing .

[![AWS ECS](https://img.shields.io/badge/AWS-ECS-orange)](https://aws.amazon.com/ecs/)
This project utilizes AWS ECS (Elastic Container Service) for running containers.

[![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-orange)](https://dvc.org/)
This project utilizes DVC (Data Version Control) for managing and versioning data sets.


[![Apache Airflow](https://img.shields.io/badge/Apache-Airflow-red)](https://airflow.apache.org/)
This project utilizes Apache Airflow to automate and orchestrate batch data processing tasks and model training.


[![Forecasting Stock Market price application](https://github.com/MSaadMakhdoom/Forecasting-Stock-Market-Prices-using-Machine-Learning/actions/workflows/main-app.yml/badge.svg)](https://github.com/MSaadMakhdoom/Forecasting-Stock-Market-Prices-using-Machine-Learning/actions/workflows/main-app.yml)
#
[![Build Docker Image](https://github.com/MSaadMakhdoom/Forecasting-Stock-Market-Prices-using-Machine-Learning/actions/workflows/docker.yml/badge.svg)](https://github.com/MSaadMakhdoom/Forecasting-Stock-Market-Prices-using-Machine-Learning/actions/workflows/docker.yml)
# Forecasting-Stock-Market-Prices-using-Machine-Learning

Forecasting Stock Market Prices using Machine Learning predict stock market prices based on historical data.

This project performs stock price prediction using machine learning techniques. It uses a dataset containing historical stock market data and trains a model to make predictions on future stock prices.

1. Data Preprocessing: The code reads the stock market data from a file and prepares it for training. It removes unnecessary information and scales the data to a specific range that the model can work with. This step ensures that the data is in a suitable format for training the model.

2. Model Training: Once the data is preprocessed, the code trains a machine learning model called LSTM (Long Short-Term Memory). LSTM is a type of neural network that can analyze sequential data, such as stock prices over time. The model is trained using the preprocessed data, and its parameters are optimized to make accurate predictions. The training process involves running the data through the model multiple times and adjusting the model's internal settings to minimize prediction errors.

3. Model Evaluation: After training, the code evaluates the trained model's performance by testing it on unseen data. It takes a portion of the preprocessed data that was not used for training and feeds it into the model to make predictions on future stock prices. The predictions are compared to the actual stock prices to measure how well the model performs. The evaluation metric used is called Root Mean Squared Error (RMSE), which quantifies the average difference between predicted and actual stock prices.

4. MLflow Integration: The code integrates with MLflow, a machine learning lifecycle management platform. It uses MLflow to log important information during the training and evaluation processes. This includes parameters used in training, metrics like loss and RMSE, and even the trained model itself. MLflow provides a centralized platform to track and manage the experiments, making it easier to reproduce and compare different models.

Overall, this code automates the stock price prediction process by handling data preprocessing, model training, evaluation, and experiment tracking using Airflow.

# Airflow
Task Definition: The code defines three tasks within the DAG: preprocess_data, train_model, and test_model. Each task represents a specific action to be performed. For example, the preprocess_data task reads and preprocesses the stock market data, the train_model task trains the machine learning model, and the test_model task evaluates the model's performance.

Task Dependency: The code establishes dependencies between tasks using arrows (>>). This means that a task can only be executed once its preceding task(s) have completed successfully. In this case, the preprocess_data task must finish before the train_model task can start, and the train_model task must finish before the test_model task can start. This ensures that the tasks are executed in the correct order.
# Docker Container 
https://hub.docker.com/r/saadmakhdoom/forecasting-stock-market-prices-using-machine-learning/tags

# Demo video
[![Video](https://github.com/MSaadMakhdoom/Forecasting-Stock-Market-Prices-using-Machine-Learning/assets/62068377/88db335c-94c9-46f3-bc6f-a598ce9bb4e7)



# Setup Instructions

## Virtual Environment

Create a virtual environment:

```
python3 -m venv venv
```

Activate the virtual environment for Macbook:

```
source venv/bin/activate
```

Install the required packages from `requirements.txt`:

```
pip install -r requirements.txt
```



## flask Application Execution

Start the flask application:

```
python app.py 
```

## DVC Setup

Install DVC and the S3 bucket remote storage:

```
pip install dvc-s3
```

Set the DVC storage name:

```
dvc remote add -d img s3://projectmlops/
```

Add the dataset images to the remote folder:

```
dvc add ./img
```

Pull data from the S3 bucket:

```
aws configure
dvc pull
```

Create and run the DVC pipeline:

```
dvc run -n model_train -d face_detection_model_svm.py -o confusion_matrix.png --no-exec python3 face_detection_model_svm.py
dvc repro
```



## MLflow Setup



MLflow installed:
```
pip install mlflow

```

Start the MLflow server:

```
mlflow server


```

MLflow UI 

```
http://127.0.0.1:5000/
```

* Once the MLflow server is running, open a new terminal window.
* In the new terminal window, navigate to the directory where your MLflow project code is located.
* Executeyour MLflow code by running the Python script:
```
python mlflow_main.py
```


* During the execution of  code, MLflow will track the parameters, metrics, and artifacts defined within your code and store them in the MLflow server.
* After your code execution is complete, open your web browser and enter the URL provided by the MLflow server (e.g., http://127.0.0.1:5000/) to access the MLflow UI.


## Airflow Setup

Install Apache Airflow:

```
pip install apache-airflow

pip install apache-airflow-providers-cncf-kubernetes
```


Airflow directory in current folder:

```
export AIRFLOW_HOME=.
```

Initialize the Airflow database:

```
airflow db init
```

```
pwd
```
check path is correct in airflow.cfg


Create an admin account:

```
airflow users create --username msaad --firstname Muhammad --lastname Saad --email msaadmakhdoom@gmail.com --role Admin --password 123456
```

Start the Airflow web server:

```
airflow webserver -p 8080
```

if show error when open airflow ui execute this command
```
airflow db reset
```
Start the Airflow  scheduler
```
airflow scheduler
```

create dag folder in 





Find the PID of the Webserver and Scheduler
```
ps aux | grep airflow

```
```
kill -9 <PID>

```
Check the scheduler status:
```
ps -ef | grep "airflow scheduler"
```

 check the details of a process:
 ```
 ps -p 8739
```
