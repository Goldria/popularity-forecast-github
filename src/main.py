from model_classifier import GitHubClassifier

if __name__ == '__main__':
    new_data = [[1, 1, 1, 1, 1, 1]]
    filename = '../training_data/data_github.csv'
    classifier = GitHubClassifier(filename)

    classifier.train_svm()

    print("Classifier Metrics:")
    print(f'Precision: {classifier.count_precision()}')
    print(f'Recall: {classifier.count_recall()}')
    print(f'F1-score: {classifier.count_f1()}')
    class_rf = classifier.predict_class(new_data)
    print(
        f'Predicted class: {class_rf}')
