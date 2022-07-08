import tensorflow as tf
from typing import List
from app.config import EVENT_PROBABILITY_THRESHOLD

from .base import BasePredictor
from shared import log_before


class EventTypesPredictor(BasePredictor):

    @log_before
    def predict_batch(self, messages) -> (List[List[str]], List[List[float]]):
        """
        :param messages: list of lmc messages
        :return: list of lists of events types + list of lists of probabilities
        """
        res = self.model(tf.convert_to_tensor(messages, dtype=tf.string))
        # log.debug(f'Inference result -----> {res}')
        str_labels = []
        probs = []
        for i, j in zip(messages, res):
            labels = tf.where(j > EVENT_PROBABILITY_THRESHOLD).numpy().flatten()
            str_labels.append([self.name_mapping[i] for i in labels])
            pr = tf.gather(j, labels) * 100
            pr = tf.reshape(pr, shape=[-1])
            pr = pr.numpy().astype(str)
            probs.append(list(pr))
        return str_labels, probs
