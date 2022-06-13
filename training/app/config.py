from os import environ

S3_BUCKET_NAME = environ.get('S3_BUCKET', '7s-track-dev-mm-ml')

MODEL_DIR = environ.get('MODEL_DIR', '/Users/artemsereda/Documents/IdeaProjects/datahub-mm-ml/saved_models')

DATA_PATH = '/Users/artemsereda/Documents/IdeaProjects/datahub-mm-ml/saved_data/v1'
