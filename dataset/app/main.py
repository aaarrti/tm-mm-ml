from __future__ import print_function, with_statement, absolute_import

import click

from data import *
from preprocessing import stop_words_removing_text_cleaner
from shared.aws.s3 import upload_folder_to_s3


def update_ds(cls):
    ds = cls(cleaner=stop_words_removing_text_cleaner)
    path = save_ds(name=ds.name,
                   train=ds.train,
                   val=ds.val,
                   class_weights=ds.class_weights,
                   label_mapping=ds.label_mapping
                   )
    upload_folder_to_s3(inputDir=path, s3Path=path)


@click.command()
@click.argument('task', nargs=1)
def main(task):
    if task == 'status':
        update_ds(ShipmentStatusDataset)
    elif task == 'return_status':
        update_ds(ReturnShipmentStatusDataset)
    elif task == 'events':
        update_ds(EventTypesDataset)
    elif task == 'return_event':
        update_ds(ReturnEventTypesDataset)
    elif task == 'all':
        [update_ds(i) for i in [
            ShipmentStatusDataset,
            ReturnShipmentStatusDataset,
            EventTypesDataset,
            ReturnEventTypesDataset
        ]]


if __name__ == '__main__':
    main()
