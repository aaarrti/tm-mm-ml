import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as tf_text # noqa
import tensorflow_model_optimization as tfmot # noqa
import tensorflow_addons as tfa

from training.app.model.parameters import TrainingHyperParameters, NNHyperParameters, clustering_params

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

    # clustered_model = tfmot.clustering.keras.cluster_weights(model, **clustering_params)

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
    # opt = create_weight_decay_optimizer(trainParameters.base_lr)
    opt = tf.keras.optimizers.Adam(trainParameters.base_lr)

    return build_lang_agnostic_model(
        num_classes=trainParameters.numClasses,
        activation=hyperParameters.ACTIVATION,
        _optimizer=opt,
        _loss=hyperParameters.LOSS,
        _metric=hyperParameters.METRIC,
        drop_out_rate=trainParameters.DROP_OUT_RATE,
        model_name=model_name
    )


def create_weight_decay_optimizer(base_lr):
    step = tf.Variable(0, trainable=False)
    schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
        [10000, 15000], [1e-0, 1e-1, 1e-2]
    )
    # lr and wd can be a function or a tensor
    lr = base_lr * schedule(step)
    wd = lambda: 1e-4 * schedule(step)
    return tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)

