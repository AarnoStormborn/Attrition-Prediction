FROM python:3.9

WORKDIR /serve

COPY ./src ./src
COPY ./config ./config
COPY ./artifacts/data_preprocessor ./artifacts/data_preprocessor
COPY ./artifacts/model_trainer ./artifacts/model_trainer
COPY requirements.txt requirements.txt
COPY setup.py setup.py

RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

CMD [ "python", "./src/api/app.py"]