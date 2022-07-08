from typing import Dict
import tensorflow as tf

from shared.util import load_pickle, log_before
from dataclasses import dataclass


@dataclass
class Dataset:
    SHUFFLE_SIZE = 100
    name: str
    num_classes: int
    class_weights: Dict[int, float]
    label_mapping: Dict[int, str]
    train: tf.data.Dataset
    val: tf.data.Dataset


def configure_ds(ds: tf.data.Dataset, bsize):
    return ds.shuffle(Dataset.SHUFFLE_SIZE).batch(bsize).cache().prefetch(tf.data.experimental.AUTOTUNE)


@log_before
def load_ds(path, name, bsize) -> Dataset:
    train = tf.data.experimental.load(path + '/train',
                                      # element_spec=(
                                      #    tf.TensorSpec(shape=(), dtype=tf.string, name=None),
                                      #    tf.TensorSpec(shape=(), dtype=tf.int32, name=None)
                                      # )
                                      )
    train = configure_ds(train, bsize)
    val = tf.data.experimental.load(path + '/val',
                                    # element_spec=(
                                    #    tf.TensorSpec(shape=(), dtype=tf.string, name=None),
                                    #    tf.TensorSpec(shape=(), dtype=tf.int32, name=None)
                                    # )
                                    )
    val = configure_ds(val, bsize)
    class_weights = load_pickle(path + '/class_weights')
    label_mapping = load_pickle(path + '/label_mapping')

    return Dataset(
        name=name,
        train=train,
        val=val,
        class_weights=class_weights,
        label_mapping=label_mapping,
        num_classes=len(class_weights.keys())
    )
