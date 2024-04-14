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
    GitHubClassifier - классификатор для прогнозирования популярности репозиториев GitHub.

    Атрибуты:
        file_path (str): Путь к набору данных.
        model_loaded (bool): Показывает, загружена ли модель или обучена заново.
        classifier (object): Обученная модель классификатора.
        scaler (object): Объект MinMaxScaler для масштабирования признаков.
        X_train (array-like): Матрица признаков обучающих данных.
        X_test (array-like): Матрица признаков тестовых данных.
        y_train (array-like): Метки целевых признаков обучающих данных.
        y_test (array-like): Метки целевых признаков тестовых данных.

    Методы:
        __init__(self, file_path):
            Инициализирует объект GitHubClassifier. Загружает предварительно обученную модель, если доступна,
            в противном случае обучает новую модель.
        train_model(self, file_path):
            Считывает набор данных, предобрабатывает его и разделяет на обучающий и тестовый наборы.
        train_random_forest(self):
            Обучает классификатор случайного леса.
        train_gradient_boosting(self):
            Обучает классификатор градиентного бустинга.
        train_adaboost(self):
            Обучает классификатор AdaBoost.
        train_naive_bayes(self):
            Обучает классификатор наивного Байеса.
        train_decision_tree(self):
            Обучает классификатор дерева решений.
        count_f1(self):
            Вычисляет F1-меру классификатора.
        count_precision(self):
            Вычисляет точность классификатора.
        count_recall(self):
            Вычисляет полноту классификатора.
        predict_class(self, new_data):
            Предсказывает класс новых данных.
        get_confusion_matrix(self):
            Вычисляет матрицу ошибок классификатора.
        plot_confusion_matrix(self, file_path):
            Строит и сохраняет матрицу ошибок.
        save_model(self, file_path):
            Сохраняет обученную модель в файл.
        load_model(self):
            Загружает предварительно обученную модель из файла.
    """

    def __init__(self, file_path):
        """
        Инициализирует объект GitHubClassifier.

        Args:
            file_path (str): Путь к набору данных.
        """
        self.file_path = file_path
        self.model_loaded = False
        if file_path.endswith('.pkl'):
            self.load_model()
            self.model_loaded = True
        else:
            self.train_model(file_path)
            self.model_loaded = True

    def train_model(self, file_path):
        """
        Считывает набор данных, предобрабатывает его и разделяет на обучающий и тестовый наборы.
        """
        data = self.get_popular_classes(pd.read_csv(file_path))
        self.X = data.drop(['repo_name', 'popularity_class'], axis=1)
        self.y = data['popularity_class']
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, self.y, test_size=0.2, random_state=42)

    def get_popular_classes(self, data):
        """
        Назначает классы популярности для переданных данных на основе накопительных метрик.

        Аргументы:
            data (DataFrame): Входные данные, содержащие метрики для репозиториев GitHub.

        Возвращает:
            DataFrame: DataFrame с входными данными, в котором добавлен дополнительный столбец 'popularity_class',
                    указывающий класс популярности, присвоенный каждому репозиторию.
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

    def train_random_forest(self):
        """
        Обучает классификатор Random Forest.
        """
        self.classifier = RandomForestClassifier(
            n_estimators=100, random_state=42)
        self.classifier.fit(self.X_train, self.y_train)

    def train_gradient_boosting(self):
        """
        Обучает классификатор Gradient Boosting.
        """
        self.classifier = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.2, random_state=42)
        self.classifier.fit(self.X_train, self.y_train)

    def train_adaboost(self):
        """
        Обучает классификатор AdaBoost.
        """
        self.classifier = AdaBoostClassifier(
            n_estimators=50, random_state=42)
        self.classifier.fit(self.X_train, self.y_train)

    def train_naive_bayes(self):
        """
        Обучает классификатор Gaussian Naive Bayes.
        """
        self.classifier = GaussianNB()
        self.classifier.fit(self.X_train, self.y_train)

    def train_decision_tree(self):
        """
        Обучает классификатор Decision Tree.
        """
        self.classifier = DecisionTreeClassifier(random_state=42)
        self.classifier.fit(self.X_train, self.y_train)

    def count_f1(self):
        """
        Вычисляет F1-меру классификатора.

        Возвращает:
            float: F1-мера.
        """
        y_pred = self.classifier.predict(self.X_test)
        return f1_score(self.y_test, y_pred, average='weighted')

    def count_precision(self):
        """
        Вычисляет точность классификатора.

        Возвращает:
            float: Точность.
        """
        y_pred = self.classifier.predict(self.X_test)
        return precision_score(self.y_test, y_pred,
                               average='weighted', zero_division=1)

    def count_recall(self):
        """
        Вычисляет полноту классификатора.

        Возвращает:
            float: Полнота.
        """
        y_pred = self.classifier.predict(self.X_test)
        return recall_score(self.y_test, y_pred,
                            average='weighted', zero_division=1)

    def predict_class(self, new_data):
        """
        Предсказывает класс новых данных.

        Args:
            new_data (array-like): Новые данные для классификации.

        Возвращает:
            array-like: Предсказанные метки классов.
        """
        new_data_scaled = self.scaler.transform(new_data)
        return self.classifier.predict(new_data_scaled)

    def get_confusion_matrix(self):
        """
        Вычисляет матрицу ошибок классификатора.

        Возвращает:
            array-like: Матрица ошибок.
        """
        y_pred = self.classifier.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        return cm

    def plot_confusion_matrix(self, file_path):
        """
        Строит и сохраняет матрицу ошибок.

        Args:
            file_path (str): Путь для сохранения графика.
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
        Сохраняет обученную модель в файл.

        Args:
            file_path (str): Путь для сохранения модели.
        """
        joblib.dump(self.classifier, file_path, compress=True)

    def load_model(self):
        """
        Загружает предварительно обученную модель из файла.
        """
        self.classifier = joblib.load(self.file_path)
