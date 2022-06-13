from cachetools import cached
from .base_ds import Dataset
from sklearn.utils import class_weight
import numpy as np

from .util import flip_dict


class ShipmentStatusDataset(Dataset):

    name = 'status'

    def _load_ds(self):
        res = [*self.mongo.message_mapping.find({'shipment_status_id': {'$exists': True}})]
        X = [i['lmc_message'] for i in res]
        ids = [i['shipment_status_id'] for i in res]
        names = [self._query_status_name_mongo(i) for i in ids]
        int_y, name_mapping = self._map_label_name_to_int(names)
        self.num_classes = len(names)
        return X, int_y, flip_dict(name_mapping)

    @cached(
        cache={},
        key=lambda self, _id: _id
    )
    def _query_status_name_mongo(self, _id: str):
        return [*self.mongo.shipment_status.find({'_id': _id})][0]['name']

    @staticmethod
    def _map_label_name_to_int(names):
        name_mapping = {}
        counter = 0
        labels = []
        for n in names:
            if n not in name_mapping:
                name_mapping[n] = counter
                counter += 1
            labels.append(name_mapping[n])
        print(f'Mapped names to labels {name_mapping}')
        return labels, name_mapping

    def _compute_class_weights(self, y_train):
        class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                          classes=np.unique(y_train),
                                                          y=y_train)
        weights = {i: class_weights[i] for i in np.unique(y_train)}
        print('Computed class weight: {}'.format(weights))
        return weights


class ReturnShipmentStatusDataset(ShipmentStatusDataset):
    name = 'return_status'

    def _load_ds(self):
        res = [*self.mongo.message_mapping.find({'return_shipment_status_id': {'$exists': True}})]
        X = [i['lmc_message'] for i in res]
        ids = [i['return_shipment_status_id'] for i in res]
        names = [self._query_status_name_mongo(i) for i in ids]
        int_y, name_mapping = self._map_label_name_to_int(names)
        self.num_classes = len(names)
        return X, int_y, flip_dict(name_mapping)
