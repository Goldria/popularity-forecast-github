from GithubClassifier import GithubClassifier

if __name__ == '__main__':
    classifier = GithubClassifier('../training_data/data_github.csv')
    classifier.train_random_forest()

    new_data = [[1, 1, 1, 1, 1]]
    stars = classifier.predict_class(new_data)
    print(f'Predicted class: {stars}')
