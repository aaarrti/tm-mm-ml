FROM tensorflow/tensorflow:2.9.1-gpu

WORKDIR /mm-ml

COPY shared shared

RUN pip install -r shared/requirements.txt

COPY training training

WORKDIR /mm-ml/training

RUN pip install -r requirements.txt

RUN mkdir /mm-ml/saved_data && mkdir /mm-ml/saved_models && \
    chmod 777 -R /mm-ml/saved_data && chmod 777 -R /mm-ml/saved_models

ENV PYTHONPATH=/mm-ml:/mm-ml/training

