# create virtual environment
conda create --name airflow_env python=3.9

# To activate this environment
conda activate airflow_env

# To deactivate an active environment
conda deactivate

 # install Apache Airflow:
pip install "apache-airflow==2.2.3" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.2.3/constraints-no-providers-3.9.txt"


# initialize the database
airflow db init


# Create an admin account:

```
airflow users create --username msaad --firstname Muhammad --lastname Saad --email msaadmakhdoom@gmail.com --role Admin --password 123456
```
# airflow folder in your root directory, so navigate to it:
cd ~/airflow


# First, start the Webserver in the daemon mode
airflow webserver -D

# run the Scheduler:
airflow scheduler -D

# check process is running
lsof -i tcp:8080

kill -9 <pid>


# DVC setup

pip install dvc

pip install gdrive


dvc init

dvc add PSX.csv

dvc remote add -d storage gdrive://1W4VEkftEoGvWedVK1LSUEQuwEkrxdBcJ