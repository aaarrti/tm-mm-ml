import tensorflow as tf
import tensorflow_addons as tfa

class TrainingHyperParameters:
    EPOCHS = 30
    BATCH_SIZE = 32
    DROP_OUT_RATE = 0.2
    numClasses: int
    TRAIN_CALLBACKS = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', min_delta=0.01, factor=0.2, patience=5, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001),
        tfa.callbacks.TQDMProgressBar()
    ]

    def __init__(self, num_classes: int):
        self.numClasses = num_classes


class NNHyperParameters:
    LOSS: str
    LR = 3e-5
    OPTIMIZER = tf.keras.optimizers.Adam(LR)
    METRIC: [str]
    ACTIVATION: str


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
