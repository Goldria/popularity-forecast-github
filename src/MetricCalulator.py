

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)


class MetricCalculator:
    """
    MetricCalculator - class for computing evaluation metrics of a classifier or regressor.

    Attributes:
        model (object): Trained classifier or regressor model.

    Methods:
        get_confusion_matrix():
            Computes the confusion matrix of the classifier.

        plot_confusion_matrix(file_path):
            Plots and saves the confusion matrix.

        count_f1():
            Computes the F1 score of the classifier.

        count_precision():
            Computes the precision of the classifier.

        count_recall():
            Computes the recall of the classifier.

        count_accuracy():
            Computes the accuracy of the classifier.

        count_mse():
            Computes the Mean Squared Error (MSE) on test data.

        count_mae():
            Computes the Mean Absolute Error (MAE) on test data.

        count_r2():
            Computes the R^2 score on test data.
    """

    def __init__(self, model):
        """
        Initializes the MetricCalculator object.

        Args:
            model (object): Trained classifier or regressor model.
        """
        self.model = model

    def get_confusion_matrix(self):
        """
        Computes the confusion matrix of the classifier.

        Returns:
            array-like: The confusion matrix.
        """
        y_pred = self.model.classifier.predict(self.model.X_test)
        cm = confusion_matrix(self.model.y_test, y_pred)
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
                    xticklabels=sorted(self.model.y_test.unique()),
                    yticklabels=sorted(self.model.y_test.unique()))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0.2)

    def count_f1(self):
        """
        Computes the F1 score of the classifier.

        Returns:
            float: The F1 score.
        """
        y_pred = self.model.classifier.predict(self.model.X_test)
        return f1_score(self.model.y_test, y_pred, average='weighted')

    def count_precision(self):
        """
        Computes the precision of the classifier.

        Returns:
            float: The precision.
        """
        y_pred = self.model.classifier.predict(self.model.X_test)
        return precision_score(self.model.y_test, y_pred,
                               average='weighted', zero_division=1)

    def count_recall(self):
        """
        Computes the recall of the classifier.

        Returns:
            float: The recall.
        """
        y_pred = self.model.classifier.predict(self.model.X_test)
        return recall_score(self.model.y_test, y_pred,
                            average='weighted', zero_division=1)

    def count_accuracy(self):
        """
        Computes the accuracy of the classifier.

        Returns:
            float: The accuracy.
        """
        y_pred = self.model.classifier.predict(self.model.X_test)
        return accuracy_score(self.model.y_test, y_pred, normalize=True)

    def count_mse(self):
        """
        Computes the Mean Squared Error (MSE) on test data.

        Returns:
            float: The MSE value.
        """
        y_pred = self.model.regressor.predict(self.model.X_test)
        return mean_squared_error(self.model.y_test, y_pred)

    def count_mae(self):
        """
        Computes the Mean Absolute Error (MAE) on test data.

        Returns:
            float: The MAE value.
        """
        y_pred = self.model.regressor.predict(self.model.X_test)
        return mean_absolute_error(self.model.y_test, y_pred)

    def count_r2(self):
        """
        Computes the R^2 score on test data.

        Returns:
            float: The R^2 score.
        """
        y_pred = self.model.regressor.predict(self.model.X_test)
        return r2_score(self.model.y_test, y_pred)
