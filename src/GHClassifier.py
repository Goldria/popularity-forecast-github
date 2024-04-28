import pandas as pd
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from IGHModel import IGHModel


class GHClassifier(IGHModel):
    """
    GHClassifier - class for training and predicting using a classifier model for GitHub repository popularity.

    Attributes:
        file_path (str): Path to the dataset.
        X (array-like): Feature matrix of the dataset.
        y (array-like): Target labels of the dataset.
        classifier (object): Trained classifier model.
        scaler (object): MinMaxScaler object for feature scaling.
        X_train (array-like): Feature matrix of training data.
        X_test (array-like): Feature matrix of testing data.
        y_train (array-like): Target labels of training data.
        y_test (array-like): Target labels of testing data.

    Methods:
        get_popular_classes(data):
            Assigns popularity classes to the given data based on cumulative metrics.

        predict(new_data):
            Predicts the class of new data.

        train_random_forest(n_estimators, random_state):
            Trains the Random Forest classifier.

        train_gradient_boosting(n_estimators, random_state, learning_rate):
            Trains the Gradient Boosting classifier.

        train_decision_tree(random_state):
            Trains the Decision Tree classifier.

        train_adaboost(n_estimators, random_state):
            Trains the AdaBoost classifier.

        train_naive_bayes():
            Trains the Gaussian Naive Bayes classifier.
    """

    def __init__(self, file_path, test_size=0.2, random_state=42):
        """
        Initializes the GHClassifier object.

        Args:
            file_path (str): Path to the dataset.   
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): Controls the randomness of the dataset shuffling.
        """
        self.file_path = file_path
        data = self.get_popular_classes(pd.read_csv(file_path))
        self.X = data.drop(['repo_name', 'popularity_class'], axis=1)
        self.y = data['popularity_class']
        super().train_model(test_size, random_state)

    def get_popular_classes(self, data):
        """
        Assigns popularity classes to the given data based on cumulative metrics.

        Args:
            data (DataFrame): Input data containing metrics for GitHub repositories.

        Returns:
            DataFrame: DataFrame with the input data, with an additional 'popularity_class' column
                indicating the assigned popularity class for each repository.
        """
        metrics = ['forks', 'commits', 'issues', 'commits',
                   'pull_requests', 'distributors']
        total = data[metrics].sum(axis=1)
        quantiles = total.quantile([i/10 for i in range(11)])

        def assign_popularity_class(value):
            for i, quantile in enumerate(quantiles):
                if value <= quantile:
                    return i

        data['popularity_class'] = total.apply(assign_popularity_class)

        return data

    def predict(self, new_data):
        """
        Predicts the class of new data.

        Args:
            new_data (array-like): New data for classification.

        Returns:
            array-like: Predicted class labels.
        """
        new_data_scaled = self.scaler.transform(new_data)
        return self.classifier.predict(new_data_scaled)[0]

    def train_random_forest(self, n_estimators, random_state):
        """
        Trains the Random Forest classifier.

        Args:
            n_estimators (int): The number of trees in the forest.
            random_state (int): Controls the randomness of the estimator.
        """
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state)
        self.classifier.fit(self.X_train, self.y_train)

    def train_gradient_boosting(self, n_estimators, random_state, learning_rate):
        """
        Trains the Gradient Boosting classifier.

        Args:
            n_estimators (int): The number of boosting stages.
            random_state (int): Controls the randomness of the estimator.
            learning_rate (float): Learning rate shrinks the contribution of each classifier.
        """
        self.classifier = GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
        self.classifier.fit(self.X_train, self.y_train)

    def train_decision_tree(self, random_state):
        """
        Trains the Decision Tree classifier.

        Args:
            random_state (int): Controls the randomness of the estimator.
        """
        self.classifier = DecisionTreeClassifier(random_state=random_state)
        self.classifier.fit(self.X_train, self.y_train)

    def train_adaboost(self, n_estimators, random_state):
        """
        Trains the AdaBoost classifier.

        Args:
            n_estimators (int): The maximum number of estimators at which boosting is terminated.
            random_state (int): Controls the randomness of the estimator.
        """
        self.classifier = AdaBoostClassifier(
            n_estimators=n_estimators, random_state=random_state)
        self.classifier.fit(self.X_train, self.y_train)

    def train_naive_bayes(self):
        """
        Trains the Gaussian Naive Bayes classifier.
        """
        self.classifier = GaussianNB()
        self.classifier.fit(self.X_train, self.y_train)
