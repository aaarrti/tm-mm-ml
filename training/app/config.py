from os import environ

S3_BUCKET_NAME = environ.get('S3_BUCKET', '7s-track-dev-mm-ml')

MODEL_DIR = environ.get('MODEL_DIR', '/Users/artemsereda/Documents/IdeaProjects/datahub-mm-ml/saved_models')

DATASET_TAG = environ.get('DATASET_TAG', '06.2022:10:00')

LOCAL_DATA_PATH = environ.get('LOCAL_DATA_PATH', '/saved_data')

DATASET_PATH = f'{LOCAL_DATA_PATH}/{DATASET_TAG}'
