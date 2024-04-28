import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from IGHModel import IGHModel


class GHRegressor(IGHModel):
    """
    GHRegressor - class for training and predicting using a regression model for GitHub repository stars prediction.

    Attributes:
        file_path (str): Path to the dataset file.
        X (array-like): Feature matrix of the dataset.
        y (array-like): Target labels of the dataset.
        regressor (object): Trained regression model.
        scaler (object): MinMaxScaler object for feature scaling.
        X_train (array-like): Feature matrix of training data.
        X_test (array-like): Feature matrix of testing data.
        y_train (array-like): Target labels of training data.
        y_test (array-like): Target labels of testing data.

    Methods:
        predict(new_data):
            Predicts the number of stars for new data.

        train_random_forest(n_estimators, random_state):
            Trains the Random Forest model.

        train_gradient_boosting(n_estimators, random_state, learning_rate):
            Trains the Gradient Boosting model.

        train_decision_tree_regression(random_state):
            Trains the Decision Tree regression model.

        train_linear_regression():
            Trains the Linear Regression model.
    """

    def __init__(self, file_path, test_size=0.2, random_state=42):
        """
        Initializes the GHRegressor object.

        Args:
            file_path (str): Path to the dataset file.
        """
        self.file_path = file_path
        data = pd.read_csv(file_path)
        self.X = data[['forks', 'issues', 'commits',
                       'pull_requests', 'distributors']]
        self.y = data['stars']
        super().train_model(test_size, random_state)

    def train_random_forest(self, n_estimators, random_state):
        """
        Trains the Random Forest model.

        Args:
            n_estimators (int): The number of trees in the forest.
            random_state (int): Controls the randomness of the estimator.
        """
        self.regressor = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state)
        self.regressor.fit(self.X_train, self.y_train)

    def train_gradient_boosting(self, n_estimators, random_state, learning_rate):
        """
        Trains the Gradient Boosting model.

        Args:
            n_estimators (int): The number of boosting stages.
            random_state (int): Controls the randomness of the estimator.
            learning_rate (float): Learning rate shrinks the contribution of each regressor.
        """
        self.regressor = GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
        self.regressor.fit(self.X_train, self.y_train)

    def train_decision_tree_regression(self, random_state):
        """
        Trains the Decision Tree regression model.

        Args:
            random_state (int): Controls the randomness of the estimator.
        """
        self.regressor = DecisionTreeRegressor(random_state=random_state)
        self.regressor.fit(self.X_train, self.y_train)

    def train_linear_regression(self):
        """
        Trains the Linear Regression model.
        """
        self.regressor = LinearRegression()
        self.regressor.fit(self.X_train, self.y_train)

    def predict(self, new_data):
        """
        Predicts the number of stars for new data.

        Args:
            new_data (array): New data for prediction.

        Returns:
            array: Predicted number of stars.
        """
        new_data_scaled = self.scaler.transform(new_data)
        return int(self.regressor.predict(new_data_scaled)[0])
