import logging

import tensorflow as tf
from abc import ABC, abstractmethod
from .config import *
from pathlib import Path
from typing import Dict

from shared.util import log_before, load_pickle
from shared.aws.s3 import download_dir
from shared.aws.config import S3_BUCKET_NAME, S3_MODEL_DIR_PREFIX

log = logging.getLogger(__name__)


class BasePredictor(ABC):
    name_mapping: Dict[int, str]
    model: tf.keras.Model

    @log_before
    def __init__(self, model_name: str):
        self.model = self._load_model(model_name)
        self.name_mapping = load_pickle(f'{MODEL_FS_CACHE_DIR}/{S3_MODEL_DIR_PREFIX}/{model_name}/{LABEL_MAPPING_NAME}')

    @abstractmethod
    def predict_batch(self, messages):
        pass

    @staticmethod
    @log_before
    def _load_model(model_name):
        keras_model_path = f'{MODEL_FS_CACHE_DIR}/{S3_MODEL_DIR_PREFIX}/{model_name}/model'
        if Path(keras_model_path).exists():
            log.info(f'Model already exists in {keras_model_path}')
        else:
            download_dir(prefix=S3_MODEL_DIR_PREFIX,
                         local=MODEL_FS_CACHE_DIR,
                         bucket=S3_BUCKET_NAME)
        return tf.saved_model.load(keras_model_path)
