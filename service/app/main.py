from __future__ import print_function, absolute_import, annotations, nested_scopes, with_statement
import tensorflow_text # noqa

import logging
import connexion
from concurrent.futures import ThreadPoolExecutor

from app.service import MessageListener, prediction_service
from app.config import LOG_FORMAT, THREAD_POOL_SIZE, PORT
from stubs.encoder import JSONEncoder


from shared.util import print_env


logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)
log = logging.getLogger(__name__)


def serve():
    app = connexion.App(__name__, specification_dir='./stubs/openapi/')
    app.app.json_encoder = JSONEncoder

    app.add_api('openapi.yaml',
                arguments={'title': 'MM ML'},
                pythonic_params=True)
    app.run(port=PORT, host='0.0.0.0')


if __name__ == '__main__':
    print_env()

    # construct instance before starting up application
    prediction_service()
    listener = MessageListener()

    with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as executor:
        executor.submit(serve)
        executor.submit(listener.recv)
