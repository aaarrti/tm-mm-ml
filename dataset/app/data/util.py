import tensorflow as tf
from typing import Dict
from datetime import datetime

from ..config import DATA_PATH
from shared.util import save_pickle, ensure_dir


def make_dataset(x, y):
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.constant(x, dtype=tf.string), tf.constant(y, dtype=tf.int32))
    )
    return dataset


def save_ds(name: str,
            train: tf.data.Dataset,
            val: tf.data.Dataset,
            label_mapping: Dict[int, str],
            class_weights: Dict[int, float]
            ) -> str:
    """
    :param name: name of ds
    :param train: train split
    :param val: validation split
    :param label_mapping: class label to class label mapping
    :param class_weights:
    :return: path to dir where ds is saved
    """
    print(f'Saving {name} dataset')
    ensure_dir(DATA_PATH)

    tag = datetime.now().strftime("%m.%d:%H.%M")

    ensure_dir(f'{DATA_PATH}/{tag}')
    ensure_dir(f'{DATA_PATH}/{tag}/{name}')

    tf.data.experimental.save(train, f'{DATA_PATH}/{tag}/{name}/train')
    tf.data.experimental.save(val, f'{DATA_PATH}/{tag}/{name}/val')

    save_pickle(class_weights, f'{DATA_PATH}/{tag}/{name}/class_weights')
    save_pickle(label_mapping, f'{DATA_PATH}/{tag}/{name}/label_mapping')
    return f'{DATA_PATH}/{tag}'


def flip_dict(d: Dict):
    return dict((v, k) for k, v in d.items())
