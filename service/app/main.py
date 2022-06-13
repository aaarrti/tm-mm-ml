from __future__ import print_function, absolute_import, annotations, nested_scopes, with_statement
import tensorflow_text # noqa

import logging
import os


from concurrent.futures import ThreadPoolExecutor
from shared.util import log_before

from service.listener import MessageListener
from config import *


logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)
log = logging.getLogger(__name__)

'''
@log_before
def serve():
    app = connexion.App(__name__, specification_dir='./stubs/openapi/')
    app.app.json_encoder = encoder.JSONEncoder
    app.add_api('openapi.yaml',
                arguments={'title': 'MM ML'},
                pythonic_params=True)
    app.run(port=PORT, host='0.0.0.0')
'''


if __name__ == '__main__':
    [log.debug(f'{i} ---> {os.environ[i]}') for i in os.environ]

    listener = MessageListener()

    with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as executor:
        executor.submit(listener.recv)
        # executor.submit(serve)
