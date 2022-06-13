from sklearn.utils import compute_class_weight
import numpy as np
import tensorflow as tf
from typing import List

from .base_ds import Dataset
from .util import flip_dict


class EventTypesDataset(Dataset):
    name = 'events'

    def _load_ds(self):
        res = [*self.mongo.message_mapping.find({'$where': 'this.event_type_ids.length > 0'})]
        X = [i['lmc_message'] for i in res]
        Y = [i['event_type_ids'] for i in res]
        int_y, name_mapping = self._map_to_int_labels(Y)
        self.Y = int_y # leave it, it needed for class weights later
        multi_hot_y = self._binarize_labels(int_y, list(name_mapping.values()))
        return X, multi_hot_y, flip_dict(name_mapping)

    @staticmethod
    def _map_to_int_labels(names: [str]):
        name_map = {}
        int_labels = []
        label_generator = 0
        for i in names:
            entry = []
            for j in i:
                if j not in name_map:
                    name_map[j] = label_generator
                    label_generator += 1
                entry.append(name_map[j])
            int_labels.append(entry)
        print(f'Mapped names to labels {name_map}')
        return int_labels, name_map

    @staticmethod
    def _binarize_labels(labels, vocab: List[int]):
        lookup = tf.keras.layers.IntegerLookup(output_mode="multi_hot", vocabulary=vocab, oov_token=0)
        train_label_binarized = lookup(tf.ragged.constant(labels)).numpy()
        return train_label_binarized

    @staticmethod
    def flatten_list(t):
        return [item for sublist in t for item in sublist]

    def _compute_class_weights(self, y):
        flat_y = self.flatten_list(self.Y)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(flat_y),
                                                          y=flat_y)
        weights = {i: class_weights[i] for i in np.unique(flat_y)}
        print(f'Computed class weight: {weights}')
        return weights


class ReturnEventTypesDataset(EventTypesDataset):
    name = 'return_events'

    def _load_ds(self):
        res = [*self.mongo.message_mapping.find({'$where': 'this.return_event_type_ids.length > 0'})]
        X = [i['lmc_message'] for i in res]
        Y = [i['return_event_type_ids'] for i in res]
        int_y, name_mapping = self._map_to_int_labels(Y)
        self.Y = int_y  # leave it, it needed for class weights later
        multi_hot_y = self._binarize_labels(int_y, list(name_mapping.values()))
        return X, multi_hot_y, flip_dict(name_mapping)

