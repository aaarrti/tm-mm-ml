from __future__ import absolute_import, print_function, with_statement, annotations

import click
import tensorflow as tf
# from tensorflow.python.framework.ops import disable_eager_execution

from shared.util import ensure_dir, save_pickle
from shared.aws.s3 import upload_folder_to_s3, download_folder_from_s3

from training.app.model import *
from utils import *

from training.app.config import *
from shared.util import print_env
from shared.config import *


def train(ds: Dataset, tr: TrainingHyperParameters, nnPr):
    model_name = name_model(ds)

    ensure_dir(f'{APP_PATH}/{MODEL_DIR}/{model_name}')

    model = build_model(hyperParameters=nnPr, trainParameters=tr)
    print(f'Fine tuning {model.name}')

    history = model.fit(
        ds.train,
        validation_data=ds.val,
        epochs=tr.epochs,
        class_weight=ds.class_weights,
        callbacks=tr.TRAIN_CALLBACKS,
        verbose=2
    )
    history_dict = history.history

    tf.saved_model.save(model, f'{APP_PATH}/{MODEL_DIR}/{model_name}')

    save_pickle(ds.label_mapping, f'{APP_PATH}/{MODEL_DIR}/{model_name}/label_mapping')
    save_pickle(history_dict, f'{APP_PATH}/{MODEL_DIR}/{model_name}/history')

    upload_folder_to_s3(
        local=f'{APP_PATH}/{MODEL_DIR}/{model_name}',
        s3Path=f'{MODEL_DIR}/{model_name}'
    )


def train_outbound_events(epochs, lr, bsize):
    download_folder_from_s3(
        local=f'{APP_PATH}',
        s3_path=f'{DATASET_DIR}/{DATASET_TAG}/events'
    )
    ds = load_ds(f'{APP_PATH}/saved_data/{DATASET_TAG}/events', 'events', bsize)
    tr = TrainingHyperParameters(numClasses=ds.num_classes, epochs=epochs, base_lr=lr, batch_size=bsize)
    train(ds, tr, EventTypesNNHyperParameters())


def train_return_events(epochs, lr, bsize):
    download_folder_from_s3(
        local=f'{APP_PATH}',
        s3_path=f'{DATASET_DIR}/{DATASET_TAG}/return_events'
    )
    ds = load_ds(f'{APP_PATH}/saved_data/{DATASET_TAG}/return_events', 'return_events', bsize)
    tr = TrainingHyperParameters(numClasses=ds.num_classes, epochs=epochs, base_lr=lr, batch_size=bsize)
    train(ds, tr, EventTypesNNHyperParameters())


def train_outbound_status(epochs, lr, bsize):
    download_folder_from_s3(
        local=f'{APP_PATH}',
        s3_path=f'{DATASET_DIR}/{DATASET_TAG}/status'
    )
    ds = load_ds(f'{APP_PATH}/saved_data/{DATASET_TAG}/status', 'status', bsize)
    tr = TrainingHyperParameters(numClasses=ds.num_classes, epochs=epochs, base_lr=lr, batch_size=bsize)
    train(ds, tr, ShipmentStatusNNHyperParameters())


def train_return_status(epochs, lr, bsize):
    download_folder_from_s3(
        local=f'{APP_PATH}',
        s3_path=f'{DATASET_DIR}/{DATASET_TAG}/return_status'
    )
    ds = load_ds(f'{APP_PATH}/saved_data/{DATASET_TAG}/return_status', 'status', bsize)
    tr = TrainingHyperParameters(numClasses=ds.num_classes, epochs=epochs, base_lr=lr, batch_size=bsize)
    train(ds, tr, ShipmentStatusNNHyperParameters())


@click.command()
@click.argument('task', nargs=1, required=True, type=str)
@click.option('--epochs', default=EPOCHS)
@click.option('--base-lr', default=LR)
@click.option('--batch-size', default=BATCH_SIZE)
def main(task, epochs, base_lr, batch_size):
    if task == 'status':
        train_outbound_status(epochs, base_lr, batch_size)
    elif task == 'return-status':
        train_return_status(epochs, base_lr, batch_size)
    elif task == 'events':
        train_outbound_events(epochs, base_lr, batch_size)
    elif task == 'return-events':
        train_return_events(epochs, base_lr, batch_size)
    # crashes with OOM if u try to train 2+ models in same process
    # elif task == 'all':
    #    train_outbound_status()
    #    train_return_status()
    #    train_outbound_events()
    #    train_return_events()
    else:
        raise Exception(f'Unknown task {task}')


if __name__ == '__main__':
    # disable_eager_execution()
    print_env()
    print('-' * 50)
    print('Running TF ----> ', tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    print(f'GPUs: {gpus}')
    print('-' * 50)
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    main()
