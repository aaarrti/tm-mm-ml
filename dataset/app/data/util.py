import tensorflow as tf
from typing import Dict

from shared import save_pickle, ensure_dir, time_stamp_tag, log_before
from shared.config import *


def make_dataset(x, y):
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.constant(x, dtype=tf.string), tf.constant(y, dtype=tf.int32))
    )
    return dataset


def save_ds(name: str,
            train: tf.data.Dataset,
            val: tf.data.Dataset,
            label_mapping: Dict[int, str],
            class_weights: Dict[int, float],
            ) -> str:
    """
    :param name: name of ds
    :param train: train split
    :param val: validation split
    :param label_mapping: class label to class label mapping
    :param class_weights:
    :return: tag of ds saved
    """
    print(f'Saving {name} dataset')

    tag = time_stamp_tag()
    ensure_dir(f'{APP_PATH}/{DATASET_DIR}/{tag}/{name}/train')
    ensure_dir(f'{APP_PATH}/{DATASET_DIR}/{tag}/{name}/val')

    _save_ds(train, f'{APP_PATH}/{DATASET_DIR}/{tag}/{name}/train')
    _save_ds(val, f'{APP_PATH}/{DATASET_DIR}/{tag}/{name}/val')

    save_pickle(class_weights, f'{APP_PATH}/{DATASET_DIR}/{tag}/{name}/class_weights')
    save_pickle(label_mapping, f'{APP_PATH}/{DATASET_DIR}/{tag}/{name}/label_mapping')

    return tag


@log_before
def _save_ds(ds, path):
    tf.data.experimental.save(ds, path)


def flip_dict(d: Dict):
    return dict((v, k) for k, v in d.items())
