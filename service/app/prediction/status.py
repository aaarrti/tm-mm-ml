import tensorflow as tf
from typing import List

from .base import BasePredictor
from shared import log_before


class StatusPredictor(BasePredictor):

    @log_before
    def predict_batch(self, messages: List[str]) -> (List[str], List[float]):
        """
        :param messages: List of lmc_messages
        :return: list of statuses + list of probabilities
        """
        res = self.model(tf.convert_to_tensor(messages, dtype=tf.string))
        # log.debug(f'Inference result -----> {res}')
        probs = tf.reduce_max(res, axis=1) * 100
        labels = tf.math.argmax(res, axis=1)
        str_labels = [self.name_mapping[i] for i in labels.numpy()]
        return str_labels, probs.numpy().astype(str)
