import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor


class GithubRegressor:
    """
    GithubRegressor - class for training a machine learning model
    to predict the number of stars for GitHub repositories.

    Attributes:
        file_path (str): Path to the dataset file.
        scaler (object): MinMaxScaler object for feature scaling.
        regressor (object): Trained regression model.
        X_train (array-like): Feature matrix of training data.
        X_test (array-like): Feature matrix of testing data.
        y_train (array-like): Target labels of training data.
        y_test (array-like): Target labels of testing data.

    Methods:
        train_random_forest(n_estimators, random_state):
            Trains the Random Forest model.
        train_gradient_boosting(n_estimators, random_state, learning_rate):
            Trains the Gradient Boosting model.
        train_decision_tree_regression(random_state):
            Trains the Decision Tree regression model.
        train_linear_regression():
            Trains the Linear Regression model.
        count_mse():
            Computes the Mean Squared Error (MSE) on test data.
        count_mae():
            Computes the Mean Absolute Error (MAE) on test data.
        count_r2():
            Computes the R^2 score on test data.
        predict_stars(new_data):
            Predicts the number of stars for new data.
        save_model(file_path):
            Saves the trained model to a file.
        load_model(file_path):
            Loads a pre-trained model from a file.
    """

    def __init__(self, file_path, test_size=0.2, random_state=42):
        """
        Initializes the GithubRegressor object.

        Args:
            file_path (str): Path to the dataset file.
        """
        self.file_path = file_path
        data = pd.read_csv(file_path)
        self.X = data[['forks', 'issues', 'commits',
                       'pull_requests', 'distributors']]
        self.y = data['stars']
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, self.y, test_size=test_size, random_state=random_state)

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

    def count_mse(self):
        """
        Computes the Mean Squared Error (MSE) on test data.

        Returns:
            float: The MSE value.
        """
        y_pred = self.regressor.predict(self.X_test)
        return mean_squared_error(self.y_test, y_pred)

    def count_mae(self):
        """
        Computes the Mean Absolute Error (MAE) on test data.

        Returns:
            float: The MAE value.
        """
        y_pred = self.regressor.predict(self.X_test)
        return mean_absolute_error(self.y_test, y_pred)

    def count_r2(self):
        """
        Computes the R^2 score on test data.

        Returns:
            float: The R^2 score.
        """
        y_pred = self.regressor.predict(self.X_test)
        return r2_score(self.y_test, y_pred)

    def predict_stars(self, new_data):
        """
        Predicts the number of stars for new data.

        Args:
            new_data (array): New data for prediction.

        Returns:
            array: Predicted number of stars.
        """
        new_data_scaled = self.scaler.transform(new_data)
        return int(self.regressor.predict(new_data_scaled)[0])

    def save_model(self, file_path):
        """
        Saves the trained model to a file.

        Args:
            file_path (str): Path to save the model file.
        """
        joblib.dump(self.regressor, file_path, compress=True)

    def load_model(self, file_path):
        """
        Loads a pre-trained model from a file.

        Args:
            file_path (str): Path to the model file to load.
        """
        self.regressor = joblib.load(file_path)
