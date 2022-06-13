from os import environ


DATA_PATH = environ.get('DATA_PATH', '/saved_data')

MONGO_URI = environ['MONGO_URI']

