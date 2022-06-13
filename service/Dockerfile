FROM node as swagger

RUN apt update && apt install -y default-jre && npm install @openapitools/openapi-generator-cli -g

COPY service service

WORKDIR service

RUN openapi-generator-cli generate -i openapi/api.yml -g python-flask --additional-properties=packageName=app.stubs


FROM tensorflow/tensorflow:2.9.1
COPY --from=swagger /service /service

WORKDIR service
RUN pip install -r requirements.txt


COPY shared shared

ENV PYTHONPATH=/:/service

EXPOSE 9090
CMD [ "python3", "app/main.py" ]
