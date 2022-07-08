from __future__ import print_function, with_statement, absolute_import, annotations

import click

from data import *
from preprocessing import basic_text_cleaner
from shared.aws import *
from shared.config import *
from shared.util import print_env


def update_ds(cls):
    ds = cls(cleaner=basic_text_cleaner)

    tag = save_ds(name=ds.name,
                  train=ds.train,
                  val=ds.val,
                  class_weights=ds.class_weights,
                  label_mapping=ds.label_mapping,
                  )

    upload_folder_to_s3(
        local=f'{APP_PATH}/{DATASET_DIR}/{tag}/{ds.name}',
        s3Path=f'{DATASET_DIR}/{tag}/{ds.name}'
    )


TASKS = {
    'status': [ShipmentStatusDataset],
    'return-status': [ReturnShipmentStatusDataset],
    'events': [EventTypesDataset],
    'return-events': [ReturnEventTypesDataset],
    'all': [ShipmentStatusDataset, ReturnShipmentStatusDataset, EventTypesDataset, ReturnEventTypesDataset]
}


@click.command()
@click.argument('task', nargs=1)
def main(task):
    args = TASKS[task]
    for i in args:
        update_ds(i)


if __name__ == '__main__':
    print_env()
    main()
