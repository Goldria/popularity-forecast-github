import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class GitHubClassifier:
    def __init__(self, file_path):
        self.file_path = file_path
        data = pd.read_csv(file_path)
        self.X = data.drop(['repo_name'], axis=1)
        data['stars_category'] = pd.qcut(data['stars'], q=10, labels=[
                                         'Класс 1',
                                         'Класс 2',
                                         'Класс 3',
                                         'Класс 4',
                                         'Класс 5',
                                         'Класс 6',
                                         'Класс 7',
                                         'Класс 8',
                                         'Класс 9',
                                         'Класс 10'])
        self.y = data['stars_category']
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, self.y, test_size=0.2, random_state=42)

    def train_random_forest(self):
        self.classifier = RandomForestClassifier(
            n_estimators=100, random_state=42)
        self.classifier.fit(self.X_train, self.y_train)

    def train_gradient_boosting(self):
        self.classifier = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.2, random_state=42)
        self.classifier.fit(self.X_train, self.y_train)

    def count_f1(self):
        y_pred = self.classifier.predict(self.X_test)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        return f1

    def count_precision(self):
        y_pred = self.classifier.predict(self.X_test)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        return precision

    def count_recall(self):
        y_pred = self.classifier.predict(self.X_test)
        recall = recall_score(self.y_test, y_pred, average='weighted')
        return recall

    def predict_class(self, new_data):
        new_data_scaled = self.scaler.transform(new_data)
        predicted_class = self.classifier.predict(new_data_scaled)
        return predicted_class

    def get_confusion_matrix(self):
        y_pred = self.classifier.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        return cm

    def plot_confusion_matrix(self, file_path):
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
        joblib.dump(self.classifier, file_path, compress=True)

    def load_model(self, file_path):
        self.classifier = joblib.load(file_path)
