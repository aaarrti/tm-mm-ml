import logging
import time

from app.sqs import *
from app.stubs.models.lmc_message_in import LmcMessageIn
from app.config import SQS_POLL_INTERVAL_SEC
from .prediction_svc import prediction_service


log = logging.getLogger(__name__)


class MessageListener:

    def __init__(self):
        self.in_q = in_queue()
        self.out_q = out_queue()

    def recv(self):
        log.info("-----> Waiting for SQS messages")
        while True:
            try:
                messages = receive_messages(self.in_q)
                if len(messages) == 0:
                    log.info('No messages in SQS')
                    continue

                serialized_ins = [LmcMessageIn.from_dict(i) for i in messages]
                pr_svc = prediction_service()
                result = pr_svc.predict_mappings(serialized_ins)
                send_messages(self.out_q, result)
            except Exception as e:
                log.error(f'Exception while processing messages from SQS {e}')
            time.sleep(SQS_POLL_INTERVAL_SEC)
