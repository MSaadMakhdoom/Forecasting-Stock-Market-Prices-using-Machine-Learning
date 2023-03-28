FROM python:3.10
WORKDIR /app  # first set the working directory
# copy requirements.txt to current working directory
COPY requirements.txt ./  
# install required pkgs
RUN pip install -r requirements.txt
# copy other code
COPY . ./


CMD ["python", "app.py"]

# CMD gunicorn --bind 0.0.0:$PORT main:app

