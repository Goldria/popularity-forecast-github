from GithubRegressor import GithubRegressor

if __name__ == '__main__':
    filename = '../models_regression/random_forest.pkl'
    classifier = GithubRegressor('../training_data/data_github.csv')
    classifier.train_random_forest()
    classifier.save_model(filename)

    new_data = [[1, 1, 1, 1, 1]]
    stars = classifier.predict_stars(new_data)
    print(f'Predicted stars: {stars}')
