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
    GitHubRegressor - класс для обучения модели машинного обучения
    для прогнозирования количества звезд у репозитория на GitHub.

    Methods:
        train_model(file_path):
            Считывает набор данных, предобрабатывает его и разделяет на обучающий и тестовый наборы.
        train_random_forest():
            Обучает модель случайного леса.
        train_gradient_boosting():
            Обучает модель градиентного бустинга.
        train_decision_tree_regression():
            Обучает модель регрессии дерева решений.
        train_linear_regression():
            Обучает линейную регрессию.
        count_mse():
            Вычисляет MSE на тестовых данных.
        count_mae():
            Вычисляет MAE на тестовых данных.
        count_r2():
            Вычисляет R^2 на тестовых данных.
        predict_stars(new_data):
            Прогнозирует количество звезд для новых данных.
        save_model(file_path):
            Сохраняет обученную модель в файл.
        load_model():
            Загружает предварительно обученную модель из файла.
    """

    def __init__(self, file_path):
        """
        Инициализирует объект GitHubRegressor.

        Args:
            file_path (str): Путь к файлу набора данных.
        """
        self.file_path = file_path
        self.model_loaded = False
        self.regressor = None
        self.scaler = None
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
        data = pd.read_csv(file_path)
        self.X = data[['forks', 'issues', 'commits',
                       'pull_requests', 'distributors']]
        self.y = data['stars']
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, self.y, test_size=0.2, random_state=42)

    def train_random_forest(self):
        """
        Обучает модель случайного леса.
        """
        self.regressor = RandomForestRegressor(
            n_estimators=100, random_state=42)
        self.regressor.fit(self.X_train, self.y_train)

    def train_gradient_boosting(self):
        """
        Обучает модель градиентного бустинга.
        """
        self.regressor = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42)
        self.regressor.fit(self.X_train, self.y_train)

    def train_decision_tree_regression(self):
        """
        Обучает модель регрессии дерева решений.
        """
        self.regressor = DecisionTreeRegressor(random_state=42)
        self.regressor.fit(self.X_train, self.y_train)

    def train_linear_regression(self):
        """
        Обучает линейную регрессию.
        """
        self.regressor = LinearRegression()
        self.regressor.fit(self.X_train, self.y_train)

    def count_mse(self):
        """
        Вычисляет MSE на тестовых данных.

        Returns:
            float: Значение MSE.
        """
        y_pred = self.regressor.predict(self.X_test)
        return mean_squared_error(self.y_test, y_pred)

    def count_mae(self):
        """
        Вычисляет MAE на тестовых данных.

        Returns:
            float: Значение MAE.
        """
        y_pred = self.regressor.predict(self.X_test)
        return mean_absolute_error(self.y_test, y_pred)

    def count_r2(self):
        """
        Вычисляет R^2 на тестовых данных.

        Returns:
            float: Значение R^2.
        """
        y_pred = self.regressor.predict(self.X_test)
        return r2_score(self.y_test, y_pred)

    def predict_stars(self, new_data):
        """
        Прогнозирует количество звезд для новых данных.

        Args:
            new_data (array): Новые данные для прогнозирования.

        Returns:
            array: Прогнозируемое количество звезд.
        """
        new_data_scaled = self.scaler.transform(new_data)
        return self.regressor.predict(new_data_scaled)

    def save_model(self, file_path):
        """
        Сохраняет обученную модель в файл.

        Args:
            file_path (str): Путь к файлу для сохранения модели.
        """
        joblib.dump(self.regressor, file_path, compress=True)

    def load_model(self):
        """
        Загружает предварительно обученную модель из файла.
        """
        self.regressor = joblib.load(self.file_path)
