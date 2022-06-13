from abc import abstractmethod, ABC
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from typing import List

from preprocessing import TextCleaner
from .util import *
from .config import MONGO_URI


class Dataset(ABC):
    name: str
    mongo = MongoClient(MONGO_URI).mapping

    def __init__(self, cleaner: TextCleaner):
        print(f'Creating {self.name} dataset from "MongoDB", cleaner = {cleaner.name if cleaner else None}')
        self.cleaner = cleaner
        self._prepare_ds()

    def _prepare_ds(self):
        print('-' * 20)
        X, Y, label_mapping = self._load_ds()

        classes = label_mapping.values()
        class_weights = self._compute_class_weights(Y)

        print(f'Found {len(Y)} data points belonging to {len(classes)} classes')
        if self.cleaner is not None:
            print(f'Cleaning text with: {self.cleaner.name} ')
            X = self.cleaner.preprocess_text(X)

        # split size = train=0.8, val=0.2
        X_train, X_val, y_train, y_val = train_test_split(X, Y, train_size=0.8)
        print(f'Using {len(X_train)} for training {len(X_val)} for validation')
        print('-' * 20)

        self.train = make_dataset(X_train, y_train)
        self.val = make_dataset(X_val, y_val)
        self.class_weights = class_weights
        self.classes = classes
        self.label_mapping = label_mapping

    @abstractmethod
    def _load_ds(self) -> (List[str], List[int], Dict[int, str]):
        """
        The only non-generic part for all datasets
        :return: lmc_messages, their labels (normalized), all names
        All must be returned as simple python lists
        """
        pass

    @abstractmethod
    def _compute_class_weights(self, y):
        """
        Is a bit different for multi label
        """
        pass


