# Feature Extraction

We have static data about stores and time series data about sales and information about the respective dates. 
The target is to predict sales for a given store and a given date.

### Store

The `store.csv` has the following features:

| Raw Feature                   | Values       | Values missing|
|-------------------------------|---------------|-----------|
| Store                         | int ID 1-1115 | - |
| StoreType                     | {a,b,c,d}     | - |
| Assortment                    | {a,b,c}     | - |
| CompetitionDistance           | float | 3 |
| CompetitionOpenSinceMonth     | {1,...,12} | 354 |
| CompetitionOpenSinceYear      | int year (eg 2008) | 354 |
| Promo2                        | {0,1} | - |
| Promo2SinceWeek              | {1,...,50 [, ..., 52 ]} | 544 |
| Promo2SinceYear               | int year | 544 |
| PromoInterval     | str list of months abbrevations | 544 |

In order to predict sales for a given day, the relevant information seems to be the time since the nearest competition has opened (rather than the absolute date), and the time since a store is participating in a promo, and the time since the promo started last. So for any given date, these features can be calculated.
Since there are so many stores, one hot encoding of the ID is not suited. One approach can be to not use the store id directly, but rather to use the store features in order to make predictions.


The features we extract from this for prediction given `storeID` and `currDate` will then be

| Extracted Feature | Representation | Raw features used | Values missing |
|-------------------|----------------|------------------|-----------------|
| Store Type | One hot 4 | Store.StoreType | - |
| Assortment | One hot 3 | Store.Assortment | - |
| CompetitionDistance | float | Store.CompetitionDistance | 3 |
| CompetitionOpenSinceDays | uint | Store.CompetitionOpenSinceMonth, CompetitionOpenSinceYear, `currDate` | 354 |
| PromoSinceDays | uint if participating in promo, else -1 | Store.Promo2, Store.Promo2SinceWeek, Store.Promo2SinceYear, `currDate` | - | 
| DaysSinceInterval | uint if participating in promo, else -1 | `currDate`, Store.PromoInterval | 

We still have to deal with two columns that contain missing values. We can try our forecasters with different methods to fill them,
eg

| Feature | Method to fill | Rational |
|---|---|----|
| CompetitionDistance | max value | If there is no data about competitors, they might be too far away |
| CompetitionDistance | avg value | It's a typical distance of one store to the next competitor |
| CompetitionOpenSinceDays | max value | If this is unknown, the competitors may simply have been there for ages |
| CompetitionOpenSinceDays | avg value | It's a typical value  |

### Time Series Data

A summary of `train.csv` is given here (there are no values missing):

| Feature | Values | Encoding |
|---------|--------|----------|
| Store   | StoreID| - 
| Day of Week | {1, ..., 7} | one hot 7 |
| Date | YYYY-MM-DD Date format str | int days since some starting point | 
| Sales | int | int |
| Customers | int | int |
| Open | {0,1} | {0,1} |
| Promo | {0,1} |  {0,1} |
| StateHoliday | {0, 'b', 'a', '0', 'c'} | one hot 4 |
| SchoolHoliday | {0,1} |  {0,1} |

# Code structure

A description of what happens where

- `main.py` cli of the app. Argparse happens here, you can add new commands (for model selection etc) here
- `data` package for data and feature extraction ( `class DataExtraction`) and the `class Data` for these tasks:
    - Split the data in test/validatation/train data
    - next_train_batch() gives the next train batch, this should now always return a list of data of only one store, and for consecutive dates. As I mentioned in #17 your function should also add as feature the sales of the last days, so it will be next_train_batch(forecaster). Validation and test data can no longer be passed as single batch, because the network can only process one time series at a time. The solution that comes to my mind is to implement validation_batches(forecaster) that give a list of time series like [(X_1, y_1), (X_2, y_2), ...]
    - return only data from one store if specified to do so in the constructor
- `forecaster` package
...

# Visualize predictions
The following example loads a model and creates some default plots for the trained model:
- Avg prediction and error per day
- predictions and error for a random store

```python
    with tf.Session() as sess:
        model = FeedForwardNN1(sess=sess,
                                plot_dir=src_dir + "/../plots/example-model",
                                features_count=25)
        model.load_params("models/<some_model>_params")

        visualize_predictions(model, src_dir + "/../plots/example-model")
```

To add plots for other subsets of rows, have a look at how the plots for the random store are generated: (taken from `visualize_predictions.py`)

```python
    # plot random store
    data = TimeSeriesData()
    store_id = np.random.randint(1, data.store_count)
    row_ids = data.train.index[data.Store == store_id]
    plot_rows(data=data, forecaster=forecaster, row_ids=row_ids, name="Store-{}".format(store_id), output_dir=output_dir)
```

## Recurrent Neural Network
- What exactly are the time series? (One for each store?)
- How will the remaining features be integrated?
- Do we have regular intervals of data? If not, is that a problem?

## Standard Regression Model
This approach is pretty straight forward, for a given date and store, all the features can be extracted and then be fed into any regression model
Standard regression models are for example:
- Linear models
- Feed Forward Neural Networks
- SVMs
