from os import environ


SQS_BATCH_SIZE = 10

SQS_MESSAGE_ATTRIBUTES = {
    'Author': {
        'StringValue': 'MM ML',
        'DataType': 'String'
    }
}

SQS_MESSAGE_WAIT_TIME_SEC = 10


IN_QUEUE_NAME = environ.get('IN_QUEUE_NAME')
OUT_QUEUE_NAME = environ.get('OUT_QUEUE_NAME')

USE_ELASTIC_MQ = environ.get('USE_ELASTIC_MQ', default=False)
