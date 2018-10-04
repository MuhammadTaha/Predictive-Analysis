from src.data.feature_enum import FEATURES

try:
    from src.data.data_extraction import DataExtraction, DATA_DIR, DATA_PICKLE_FILE
except ModuleNotFoundError:
    from .data_extraction import DataExtraction
    from .data_extraction import DATA_DIR, DATA_PICKLE_FILE


class FeedForwardData(DataExtraction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epochs = 0
        self.used_this_epoch = set()
        self.train_point_ids = set()
        self.test_point_ids = set()
        self.is_new_epoch = None

    def train_test_split(self, train_point_ids, test_point_ids):
        self.train_point_ids, self.test_point_ids = list(train_point_ids), list(test_point_ids)

    def all_train_data(self):
        return self.data[FEATURES].iloc[self.train_point_ids].values, self.data.Sales.iloc[self.train_point_ids].values

    def all_test_data(self):
        return self.data[FEATURES].iloc[self.test_point_ids].values, self.data.Sales.iloc[self.test_point_ids].values

    def new_epoch(self):
        self.epochs += 1
        self.used_this_epoch = set()
        self.is_new_epoch = True
        print("START EPOCH", self.epochs)
