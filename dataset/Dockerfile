FROM tensorflow/tensorflow:2.9.1

WORKDIR /mm-ml

COPY shared shared

RUN pip install -r shared/requirements.txt

COPY dataset dataset

WORKDIR /mm-ml/dataset

RUN pip install -r requirements.txt

RUN mkdir /mm-ml/saved_data && chmod 777 /mm-ml/saved_data

ENV PYTHONPATH=/mm-ml:/mm-ml/dataset
