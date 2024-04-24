import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


class GithubClassifier:
    """
    GithubClassifier - classifier for predicting GitHub repository popularity.

    Attributes:
        file_path (str): Path to the dataset or trained model.
        classifier (object): Trained classifier model.
        scaler (object): MinMaxScaler object for feature scaling.
        X_train (array-like): Feature matrix of training data.
        X_test (array-like): Feature matrix of testing data.
        y_train (array-like): Target labels of training data.
        y_test (array-like): Target labels of testing data.

    Methods:
        train_random_forest(n_estimators, random_state):
            Trains the Random Forest classifier.
        train_gradient_boosting(n_estimators, random_state, learning_rate):
            Trains the Gradient Boosting classifier.
        train_decision_tree(random_state):
            Trains the Decision Tree classifier.
        train_adaboost(n_estimators, random_state):
            Trains the AdaBoost classifier.
        train_naive_bayes(:
            Trains the Gaussian Naive Bayes classifier.
        count_f1():
            Computes the F1 score of the classifier.
        count_precision():
            Computes the precision of the classifier.
        count_recall():
            Computes the recall of the classifier.
        predict_class(new_data):
            Predicts the class of new data.
        get_confusion_matrix():
            Computes the confusion matrix of the classifier.
        plot_confusion_matrix(file_path):
            Plots and saves the confusion matrix.
        save_model(file_path):
            Saves the trained model to a file.
        load_model():
            Loads a pre-trained model from a file.
    """

    def __init__(self, file_path, test_size=0.2, random_state=42):
        """
        Initializes the GithubClassifier object.

        Args:
            file_path (str): Path to the dataset.   
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): Controls the randomness of the dataset shuffling.
        """
        self.file_path = file_path
        data = self.get_popular_classes(pd.read_csv(file_path))
        self.X = data.drop(['repo_name', 'popularity_class'], axis=1)
        self.y = data['popularity_class']
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, self.y, test_size=test_size, random_state=random_state)

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

    def count_f1(self):
        """
        Computes the F1 score of the classifier.

        Returns:
            float: The F1 score.
        """
        y_pred = self.classifier.predict(self.X_test)
        return f1_score(self.y_test, y_pred, average='weighted')

    def count_precision(self):
        """
        Computes the precision of the classifier.

        Returns:
            float: The precision.
        """
        y_pred = self.classifier.predict(self.X_test)
        return precision_score(self.y_test, y_pred,
                               average='weighted', zero_division=1)

    def count_recall(self):
        """
        Computes the recall of the classifier.

        Returns:
            float: The recall.
        """
        y_pred = self.classifier.predict(self.X_test)
        return recall_score(self.y_test, y_pred,
                            average='weighted', zero_division=1)

    def predict_class(self, new_data):
        """
        Predicts the class of new data.

        Args:
            new_data (array-like): New data for classification.

        Returns:
            array-like: Predicted class labels.
        """
        new_data_scaled = self.scaler.transform(new_data)
        return self.classifier.predict(new_data_scaled)[0]

    def get_confusion_matrix(self):
        """
        Computes the confusion matrix of the classifier.

        Returns:
            array-like: The confusion matrix.
        """
        y_pred = self.classifier.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        return cm

    def plot_confusion_matrix(self, file_path):
        """
        Plots and saves the confusion matrix.

        Args:
            file_path (str): Path to save the plot.
        """
        cm = self.get_confusion_matrix()
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=sorted(self.y_test.unique()),
                    yticklabels=sorted(self.y_test.unique()))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0.2)

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
