import logging
import time


from sqs.sqs import *
from .prediction_svc import prediction_svc
from stubs.models.lmc_message_in import LmcMessageIn

log = logging.getLogger(__name__)


class MessageListener:

    def __init__(self):
        self.in_q = in_queue()
        self.out_q = out_queue()

    def recv(self):
        log.info("-----> Waiting for SQS messages")
        while True:
            messages = receive_messages(self.in_q)
            serialized_ins = [LmcMessageIn.from_dict(i) for i in messages]
            result = prediction_svc.predict_mappings(serialized_ins)
            send_messages(self.out_q, result)
            time.sleep(10)
