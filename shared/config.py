from os import environ

S3_BUCKET_NAME = environ.get('S3_BUCKET_NAME', '7s-track-dev-mm-ml')

AWS_ROLE_NAME = environ.get('AWS_ROLE_NAME', 'DataAnalyst-7s-dev-track')


# APP_PATH = environ.get('APP_PATH', '/Users/artemsereda/Documents/IdeaProjects/datahub-mm-ml')

APP_PATH = environ.get('APP_PATH', '/mm-ml')


DATASET_DIR = environ.get('DATASET_DIR', 'saved_data')

MODEL_DIR = 'saved_models'
