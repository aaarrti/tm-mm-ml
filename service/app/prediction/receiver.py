import logging
import time
from typing import List
from datetime import datetime, timedelta

from .service import PredictionService
from app.aws import *
from app.util import log_before
from app.stubs.models.lmc_message_in import LmcMessageIn
from .config import *

log = logging.getLogger(__name__)


class MessageBuffer:
    _message_buffer: List[LmcMessageIn]

    def __init__(self):
        self._message_buffer = []
        self._last_taken_at = datetime.now()

    def add(self, messages: List[LmcMessageIn]):
        [self._message_buffer.append(i) for i in messages]

    def should_take(self):
        if len(self._message_buffer) == 0:
            return False
        return len(self._message_buffer) >= BATCH_SIZE or \
               self._last_taken_at < datetime.now() - timedelta(minutes=BUFFER_IDLE_THRESHOLD_MIN)

    def take_batch(self) -> List[LmcMessageIn]:
        batch = self._message_buffer[:BATCH_SIZE]
        self._message_buffer = self._message_buffer[BATCH_SIZE:]
        self._last_taken_at = datetime.now()
        return batch

    def __len__(self):
        return len(self._message_buffer)


class MessageReceiver:

    def __init__(self):
        self.pr_svc = PredictionService()
        self.in_queue = in_queue()
        self.out_queue = out_queue()
        self.buffer = MessageBuffer()

    @log_before
    def recv(self):
        log.info("-----> Waiting for SQS messages")
        while True:
            log.info(f'<------ {len(self.buffer)} messages in buffer')
            messages = receive_messages(self.in_queue)
            self.buffer.add([LmcMessageIn.from_dict(i) for i in messages])
            if not self.buffer.should_take():
                time.sleep(POLL_INTERVAL_SEC)
                continue

            batch = self.buffer.take_batch()
            res = self.pr_svc.predict_mappings(batch)

            send_messages(self.out_queue, res)
            log.info(f'<------ {len(self.buffer)} messages in buffer')
