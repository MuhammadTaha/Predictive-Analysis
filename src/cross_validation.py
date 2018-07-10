import zipfile
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import datasets
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Activation, KerasClassifier
from sklearn.model_selection import cross_val_score
''''''
class Data():

    def __init__(self, dir="data", p_train=0.6, p_val=0.2, p_test=0.2, toy=False):
        self.data_dir = dir
        assert p_train + p_val + p_test == 1
        self.p_train, self.p_val, self.p_test = p_train, p_val, p_test

        if set(os.listdir(dir)) >= set(["sample_submission.csv", "store.csv", "test.csv", "train.csv"]):
             print("Data is extracted already")
        else:
            Data.extract(dir + "/data.zip", dir)

        self.store = pd.read_csv(dir + "/store.csv")
        self.final_test = pd.read_csv(dir + "/test.csv")
        self.train = pd.read_csv(dir + "/train.csv")

        #store = pd.read_csv(dir + "/store.csv")
        #store = store.load_store()
        #store.data[0:1]

        '''X = self.store.iloc[:, [2, 3]].values
        y = self.store.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=1000)
        
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
'''
        store = datasets.load_store()
        X = store.data
        #Y = to_categorical.store.target()
        Y = store.target

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.25, random_state=1000)
global feature_constants
def create_model(optimizer='rmsprop'):
    model = Sequential()

    model.compile(optimizer='rmsprop',
                  loss='categorical',
                  metrics=['accuracy'])
    return model

'''
#LSTM
    model = Sequential()
    model.add(Embedding(max_features, 256, input_length=maxlen))
    model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_cross_entropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=16, nb_epoch=10)
    score = model.evaluate(X_test, Y_test, batch_size=16)


    model = KerasClassifier(build_fn=create_model,
                        epochs=10,
                        batch_size=5,
                        verbose=0)

    results = cross_val_score(model, X_train, Y_train, scoring='precision_macro')

    param_grid = {'optimizer':('','')}
    grid = GridSearchCV(model,
                    param_grid=param_grid,
                    return_train_score=True,
                    scoring=['accuracy','precision_macro','recall_macro'],
                    refit='precision_macro')

    grid_results = grid.fit(X_train,Y_train)

'''
def cross_validate(model, features_data, classification_data, n_folds):
    """
    Cross-validate the given model using n_folds folds.
    """
    scores = cross_validation.cross_val_score(
        model, features_data, classification_data,
        cv=n_folds
    )
    return {
        'mean': scores.mean(),
        'sd': scores.std()
    }

def plot_accuracy_vs_folds(feature_data, classification_data_numerical):

    feature_data_train, _, classification_data_train, _ = \
        train_test_split(feature_data, classification_data_numerical)

    nbayes = GaussianNB().fit(feature_data_train, classification_data_train)
    knn3 = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn15 = neighbors.KNeighborsClassifier(n_neighbors=3)

    models = [nbayes, knn3, knn15]
    model_names = ['Naive Bayes', '3 nearest neighbour', '15 nearest neighbour']


    mean_accuracies = {}
    fold_range = range(2, 21)

    for model, model_name in zip(models, model_names):
        for n_folds in fold_range:
            scores = cross_validate(
                model,
                feature_data_train, classification_data_train,
                n_folds
            )
            if model_name in mean_accuracies.keys():
                mean_accuracies[model_name] = []
            mean_accuracies[model_name].append(scores['mean'])

    for model_name in model_names:
        plt.plot(fold_range, mean_accuracies[model_name], label=model_name)
    plt.legend(loc='best')
    plt.ylim(0.5, 1)
    plt.title("Mean Accuracy of Model for Different Numbers of Folds")
    print("For this data set and this set of models, the accuracy changes\n"
          "very little with differing numbers of folds.\n"
          "This indicates good generalisation of the models.")
    plt.show()

def optimise_knn_parameters(feature_data, classification_data_numerical):
    """ Find the set of parameters for a k-nearest neighbour classifier that yields
    the best accuracy.
    """
    feature_data_train, _, classification_data_train, _ = \
        train_test_split(feature_data, classification_data_numerical)

    parameters = [{
        'n_neighbors': [1, 3, 5, 10, 50, 100],
        'weights': ['uniform', 'distance']
    }]
    n_folds = 10

    clf = GridSearchCV(
        neighbors.KNeighborsClassifier(), parameters, cv=n_folds,
        scoring="f1" # f1 = standard measure of model accuracy
    )
    clf.fit(feature_data_train, classification_data_train)

    print("\nThe grid search scores for a k-nearest neighbour classifier were:")
    for params, mean_score, scores in clf.grid_scores_:
        print("%.1f (+-%0.03f s.d.) for %r" % (100*mean_score, scores.std()/2, params))

    print("The best parameter set found was:\n", clf.best_estimator_)

def main():
    """ Main function of the script.
    """
    feature_data, classification_data = store.load_data_set()

    # scikit-learn functions require classification in terms of numerical
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(classification_data)
    classification_data_numerical = label_encoder.transform(classification_data)

    #plot_accuracy_vs_folds(feature_data, classification_data_numerical)
    optimise_knn_parameters(feature_data, classification_data_numerical)

main()