FROM tensorflow/tensorflow:2.9.1-gpu

COPY . .
WORKDIR /app

RUN pip install -r requirements.txt

CMD ['python3', 'main.py', 'all']
