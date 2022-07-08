from os import environ

LOG_FORMAT = "%(asctime)s:%(funcName)s():%(levelname)s: %(message)s"

PORT = 9090

THREAD_POOL_SIZE = 3

SQS_BATCH_SIZE = 10

SQS_MESSAGE_ATTRIBUTES = {
    'Author': {
        'StringValue': 'MM ML',
        'DataType': 'String'
    }
}

SQS_MESSAGE_WAIT_TIME_SEC = 10

IN_QUEUE_NAME = environ.get('IN_QUEUE_NAME', 'dev_mapping_prediction_in')
OUT_QUEUE_NAME = environ.get('OUT_QUEUE_NAME', 'dev_mapping_prediction_out')

USE_ELASTIC_MQ = environ.get('USE_ELASTIC_MQ', False)

EVENT_PROBABILITY_THRESHOLD = 0.8

SHIPMENT_STATUS_MODEL_TAG = 'status_la_v1'

RETURN_SHIPMENT_STATUS_MODEL_TAG = 'return_status_la_v1'

EVENT_TYPES_MODEL_TAG = 'events_la_v1'

RETURN_EVENT_TYPES_MODEL_TAG = 'return_events_la_v1'

SQS_POLL_INTERVAL_SEC = 10
