import logging


import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Dict

from shared.util import log_before, load_pickle
from shared.aws.s3 import download_folder_from_s3
from shared.config import *

log = logging.getLogger(__name__)


class BasePredictor(ABC):
    name_mapping: Dict[int, str]
    model: tf.keras.Model

    @log_before
    def __init__(self, model_tag: str):
        self.model = self._load_model(model_tag)
        self.name_mapping = load_pickle(f'{APP_PATH}/{MODEL_DIR}/{model_tag}/label_mapping')

    @abstractmethod
    def predict_batch(self, messages):
        pass

    @staticmethod
    @log_before
    def _load_model(model_tag):
        download_folder_from_s3(
            s3_path=f'{MODEL_DIR}/{model_tag}',
            local=f'{APP_PATH}'
        )
        return tf.saved_model.load(f'{APP_PATH}/{MODEL_DIR}/{model_tag}/model')
