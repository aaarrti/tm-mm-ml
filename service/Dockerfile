FROM openapitools/openapi-generator-cli:v6.0.0 as swagger

WORKDIR /mm-ml

COPY service service

WORKDIR /mm-ml/service

RUN java -jar /opt/openapi-generator/modules/openapi-generator-cli/target/openapi-generator-cli.jar generate \
    -i openapi/api.yml -g python-flask --additional-properties=packageName=stubs,pythonSrcRoot=app


FROM tensorflow/tensorflow:2.9.1

WORKDIR /mm-ml

COPY shared shared

RUN pip install -r shared/requirements.txt

COPY --from=swagger /mm-ml/service /mm-ml/service

WORKDIR /mm-ml/service

RUN pip install -r requirements.txt

RUN  mkdir /mm-ml/saved_models &&  chmod 777 -R /mm-ml/saved_models

ENV PYTHONPATH=/mm-ml:/mm-ml/service

EXPOSE 9090

CMD [ "python3", "app/main.py" ]
