import random

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
        train_point_ids, test_point_ids = set(train_point_ids), set(test_point_ids)
        assert len(train_point_ids.intersection(test_point_ids)) == 0
        self.train_point_ids, self.test_point_ids = train_point_ids, test_point_ids
        self.new_epoch(1)
        self.X_val, self.y_val = self.all_test_data()

    def next_train_batch(self, batch_size=50):
        """
        :param batch_size: Number of rows, ignored if self.is_time_series is True
        :return X: nd.array of shape (batch_size, #features)
        :return y: nd.array of shape (batch_size, 1)
        """
        if len(self.used_this_epoch) == len(self.train_point_ids):
            self.new_epoch()
        else:
            self.is_new_epoch = False
        batch_size = min(batch_size, len(self.train_point_ids) - len(self.used_this_epoch))
        point_ids = random.sample(self.train_point_ids - self.used_this_epoch, batch_size)
        self.used_this_epoch = self.used_this_epoch.union(set(point_ids))
        return self._extract_rows(point_ids)

    def all_train_data(self):
        return self._extract_rows(self.train_point_ids)

    def all_test_data(self):
        return self._extract_rows(self.test_point_ids)

    def new_epoch(self, epoch=None):
        self.epochs = epoch if epoch is not None else self.epochs + 1
        self.used_this_epoch = set()
        self.is_new_epoch = True
        print("START EPOCH", self.epochs)
