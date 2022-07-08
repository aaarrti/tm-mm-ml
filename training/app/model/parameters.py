import tensorflow as tf
import tensorflow_model_optimization as tfmot
from dataclasses import dataclass


@dataclass
class TrainingHyperParameters:
    epochs: int
    batch_size: int
    base_lr: int
    numClasses: int

    DROP_OUT_RATE = 0.2

    TRAIN_CALLBACKS = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', min_delta=0.01, factor=0.2, patience=5, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001),
        # tfa.callbacks.TQDMProgressBar()
    ]


class NNHyperParameters:
    LOSS: str
    METRIC: [str]
    ACTIVATION: str


clustering_params = {
  'number_of_clusters': 256,
  'cluster_centroids_init': tfmot.clustering.keras.CentroidInitialization.LINEAR
}


class ShipmentStatusNNHyperParameters(NNHyperParameters):
    LOSS = 'sparse_categorical_crossentropy'
    METRIC = ['accuracy']
    ACTIVATION = 'softmax'


class EventTypesNNHyperParameters(NNHyperParameters):
    LOSS = 'binary_crossentropy'
    METRIC = ['categorical_accuracy']
    ACTIVATION = 'sigmoid'


"""
Some other worth checking out are
metric: https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/MultiLabelConfusionMatrix,
        https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/F1Score
optimizer: https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdamW

This one can be used to reduce over fitting https://www.tensorflow.org/addons/tutorials/average_optimizers_callback
"""
