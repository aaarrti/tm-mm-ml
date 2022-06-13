import click
from tensorflow.python.framework.ops import disable_eager_execution

from shared.util import ensure_dir, save_pickle
from shared.aws.s3 import upload_folder_to_s3, download_dir


from model import *
from config import *
from utils.util import name_model
from utils.ds_loader import load_ds
from utils.ds_loader import Dataset


def train(ds: Dataset, tr: TrainingHyperParameters, nnPr):
    model_name = name_model(ds)
    ensure_dir(MODEL_DIR)
    ensure_dir(f'{MODEL_DIR}/{model_name}')

    model = build_model(hyperParameters=nnPr, trainParameters=tr)
    print(f'Fine tuning {model.name}')

    history = model.fit(ds.train,
                        validation_data=ds.val,
                        epochs=TrainingHyperParameters.EPOCHS,
                        class_weight=ds.class_weights,
                        callbacks=TrainingHyperParameters.TRAIN_CALLBACKS,
                        verbose=2
                        )
    history_dict = history.history

    tf.saved_model.save(model, f'{MODEL_DIR}/{model_name}/model')
    save_pickle(ds.label_mapping, f'{MODEL_DIR}/{model_name}/label_mapping')
    save_pickle(history_dict, f'{MODEL_DIR}/{model_name}/history')
    # upload_folder_to_s3()


def train_events(ds, tr: TrainingHyperParameters):
    train(ds, tr, EventTypesNNHyperParameters())


def train_outbound_events():
    ds = load_ds(DATA_PATH + '/events', 'events')
    tr = TrainingHyperParameters(ds.num_classes)
    train_events(ds, tr)


def train_return_events():
    ds = load_ds(DATA_PATH + '/return_events', 'return_events')
    tr = TrainingHyperParameters(ds.num_classes)
    train_events(ds, tr)


def train_status(ds, tr: TrainingHyperParameters):
    train(ds, tr, ShipmentStatusNNHyperParameters())


def train_outbound_status():
    ds = load_ds(DATA_PATH + '/status', 'status')
    tr = TrainingHyperParameters(ds.num_classes)
    train_status(ds, tr)


def train_return_status():
    ds = load_ds(DATA_PATH + '/return_status', 'return_status')
    tr = TrainingHyperParameters(ds.num_classes)
    train_status(ds, tr)


@click.command()
@click.argument('task', nargs=1, required=True, type=str)
def main(task):
    if task == 'status':
        train_outbound_status()
    elif task == 'return_status':
        train_return_status()
    elif task == 'events':
        train_outbound_events()
    elif task == 'return_events':
        train_return_events()
    # crashes with OOM if u try to train 2+ models in same process
    elif task == 'all':
        train_outbound_status()
        train_return_status()
        train_outbound_events()
        train_return_events()


if __name__ == '__main__':
    # disable_eager_execution()
    print('Running TF ----> ', tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    print(f'GPUs: {gpus}')
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    main()
