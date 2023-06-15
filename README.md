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
