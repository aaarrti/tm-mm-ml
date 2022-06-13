import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as tf_text # noqa

from . import TrainingHyperParameters, NNHyperParameters

LANG_AGNOSTIC_ENCODER_NAME = "https://tfhub.dev/google/LaBSE/2"
LANG_AGNOSTIC_PREPROCESS_NAME = "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2"


def build_lang_agnostic_model(num_classes, activation, _optimizer, _loss, _metric,
                              drop_out_rate: float, model_name):
    print(f'BERT model selected           : {LANG_AGNOSTIC_ENCODER_NAME}')
    print(f'Preprocess model auto-selected: {LANG_AGNOSTIC_PREPROCESS_NAME}')
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessor = hub.KerasLayer(LANG_AGNOSTIC_PREPROCESS_NAME, name='preprocessing', trainable=False)
    encoder_inputs = preprocessor(text_input)

    encoder = hub.KerasLayer(LANG_AGNOSTIC_ENCODER_NAME, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(drop_out_rate, name='dropout')(net)
    net = tf.keras.layers.Dense(num_classes, activation=activation, name='predictions')(net)

    model = tf.keras.Model(text_input, net, name=model_name)
    model.compile(
        optimizer=_optimizer,
        loss=_loss,
        metrics=_metric,
        # jit_compile=True
    )
    print('Created model: ')
    model.summary()
    return model


def build_model(hyperParameters: NNHyperParameters, trainParameters: TrainingHyperParameters, model_name='LA-BERT'):
    return build_lang_agnostic_model(
        num_classes=trainParameters.numClasses,
        activation=hyperParameters.ACTIVATION,
        _optimizer=hyperParameters.OPTIMIZER,
        _loss=hyperParameters.LOSS,
        _metric=hyperParameters.METRIC,
        drop_out_rate=trainParameters.DROP_OUT_RATE,
        model_name=model_name
    )
