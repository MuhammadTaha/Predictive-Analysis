try:
    from src.data.timeseries_data import TimeSeriesData
except ModuleNotFoundError:
    from .timeseries_data import TimeSeriesData


class PredictedTimeseriesData(TimeSeriesData):
    """
    Get predictions of a forecaster in the same way we get true data
    Can be used to train a feed forward model and then retrieve time series data for plotting
    """

    def __init__(self, true_data, forecaster):
        """

        :param true_data: Data object
        :param forecaster: str or AbstractForecaster
        """
        if isinstance(forecaster, str):
            forecaster = AbstractForecaster.load_model(forecaster)
        self.true_data, self.forecaster = true_data, forecaster

        # modified stuff from super().__init__()
        self.data_dir = true_data.data_dir
        self.p_train, self.p_test, self.p_val = 1, 0, 0

        # load into pandas
        self.store = true_data.store
        self.final_test = true_data.final_test
        self.train = true_data.train

        self.time_count = self.train.shape[0]
        self.store_count = self.store.shape[0]

        self._prepare_time_series()

    def _extract_label(self, row_id):
        # extracts the sales from the specified row
        X = self._extract_rows([row_id])
        return self.forecaster.predict(X)
