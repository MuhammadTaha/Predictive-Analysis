try:
    from src.data.data_extraction import DataExtraction
except ModuleNotFoundError:
    from .data_extraction import DataExtraction


class TimeSeriesData(DataExtraction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepare_time_series()

    def _get_time_series(self, store_id):
        """
        :param store_id: store for which the time series will be generated
        :return date: int number of days since first date of time series
        :return X: nd.array of shape (batch_size, #features)
        :return y: nd.array of shape (batch_size, 1)
        The data returned is only for the specified store, and ordered by date
        """
        row_ids = self.train.index[self.train.Store == store_id].tolist()[::-1]
        days = [self.str_to_date(d) for d in self.train.loc[row_ids]["Date"]]
        days = [(d - days[0]).days for d in days]
        X, y = self._extract_rows(row_ids)
        return days, X, y

    def _prepare_time_series(self):
        # splits the stores into train, val and test stores
        train_count = int(self.p_train * self.store_count)
        val_count = int(self.p_val * self.store_count)
        self.train_store_ids = set(range(train_count))
        self.val_store_ids = set(range(train_count, train_count + val_count))
        self.test_store_ids = set(range(train_count + val_count, self.store_count))
        self.used_this_epoch = set()

        # create val data
        pass

        # create test data
        pass

    def next_train_batch(self, batch_size=50):
        """
        :param batch_size: Number of rows, ignored if self.is_time_series is True
        :return X: nd.array of shape (batch_size, #features)
        :return y: nd.array of shape (batch_size, 1)
        Chooses the store and let's _get_time_series do the rest
        """
        if len(self.used_this_epoch) == len(self.train_store_ids):
            self._new_epoch()
        else:
            self.is_new_epoch = False
        store_id = random.sample(self.train_store_ids - self.used_this_epoch, 1)[0]
        self.used_this_epoch = self.used_this_epoch.union(set([store_id]))
        return self._get_time_series(store_id)


