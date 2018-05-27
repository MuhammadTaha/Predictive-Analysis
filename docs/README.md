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
| CompetitionOpenSinceMonth     | {0,...,12} | 354 |
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

| Feature | Values |
|---------|--------|
| Store   | StoreID|
| Day of Week | {1, ..., 7} |
| Date | YYYY-MM-DD Date format str |
| Sales | int |
| Customers | int |
| Open | {0,1} |
| Promo | {0,1} |
| StateHoliday | {0, 'b', 'a', '0', 'c'} |
| SchoolHoliday | {0,1} |

# Forecaster

Here we can collect ideas for the forecaster.

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
