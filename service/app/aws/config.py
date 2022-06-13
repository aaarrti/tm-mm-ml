from os import environ


SQS_BATCH_SIZE = 10

SQS_MESSAGE_ATTRIBUTES = {
    'Author': {
        'StringValue': 'MM ML',
        'DataType': 'String'
    }
}

SQS_MESSAGE_WAIT_TIME_SEC = 10

IN_QUEUE_NAME = environ['IN_QUEUE_NAME']
OUT_QUEUE_NAME = environ['OUT_QUEUE_NAME']


USE_ELASTIC_MQ = environ['USE_ELASTIC_MQ'] if 'USE_ELASTIC_MQ' in environ else False
