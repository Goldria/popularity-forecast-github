from abc import ABC, abstractmethod

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class IGHModel(ABC):
    """
    IGHModel - interface defining methods for training and prediction in GitHub classifier and regressor.

    Methods:
        predict(data):
            Abstract method to predict labels or values for new data.

        train_model(test_size, random_state):
            Method to split data, scale features, and prepare datasets for training.

        save_model(file_path):
            Method to save the trained model to a file.

        load_model(file_path):
            Method to load a pre-trained model from a file.
    """
    @abstractmethod
    def predict(self, data):
        """
        Abstract method to predict labels or values for new data.

        Args:
            data (array-like): New data for prediction.

        Returns:
            array-like: Predicted labels or values.
        """
        pass

    def train_model(self, test_size, random_state):
        """
        Method to split data, scale features, and prepare datasets for training.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): Controls the randomness of the dataset shuffling.
        """
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, self.y, test_size=test_size, random_state=random_state)

    def save_model(self, file_path):
        """
        Saves the trained model to a file.

        Args:
            file_path (str): Path to save the model.
        """
        joblib.dump(self.classifier, file_path, compress=True)

    def load_model(self, file_path):
        """
        Loads a pre-trained model from a file.

        Args:
            file_path (str): Path to load the model.
        """
        self.classifier = joblib.load(file_path)
