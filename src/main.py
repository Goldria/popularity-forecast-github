from model_classifier import GitHubClassifier


new_data = [[1, 1, 1, 1, 1, 1]]
filename = '../training_data/data_github.csv'

classifier = GitHubClassifier(filename)
classifier.load_model('../trained_models/random_forest.pkl')

print("Random Forest Classifier Metrics:")
print(f'Precision: {classifier.count_precision()}')
print(f'Recall: {classifier.count_recall()}')
print(f'F1-score: {classifier.count_f1()}')
class_rf = classifier.predict_class(new_data)
print(
    f'Predicted class using Random Forest Classifier: {class_rf}')
