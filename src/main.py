from GithubClassifier import GithubClassifier
from GithubRegressor import GithubRegressor

if __name__ == '__main__':
    dataset = '../training_data/data_github.csv'

    random_forest = '../trained_models/random_forest.pkl'
    classifier = GithubClassifier(dataset, 0.2, 42)
    classifier.load_model(random_forest)
    classifier.train_random_forest(100, 42)

    new_data = [[1, 2, 3, 4, 5, 6]]
    classes = classifier.predict_class(new_data)
    print(f'Predicted class: {classes}')

    decision_tree = '../trained_models/decision_tree_regression.pkl'
    regressor = GithubRegressor(dataset, 0.2, 42)
    regressor.load_model(decision_tree)
    regressor.train_decision_tree_regression(42)

    new_data = [[1, 2, 3, 4, 5]]
    stars = regressor.predict_stars(new_data)
    print(f'Predicted stars: {stars}')
