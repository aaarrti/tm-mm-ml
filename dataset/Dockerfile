FROM tensorflow/tensorflow:2.9.1

COPY dataset dataset
COPY shared shared


WORKDIR /dataset

RUN pip install -r requirements.txt

ENV PYTHONPATH=/:/dataset

CMD [ "python3", "app/main.py", "all" ]
