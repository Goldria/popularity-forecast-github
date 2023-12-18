from flask import Flask, render_template, request

from model_classifier import GitHubClassifier

app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    data_from_post = []
    repo = request.form['repo-name']
    data_from_post.append(request.form['forks'])
    data_from_post.append(request.form['stars'])
    data_from_post.append(request.form['issues'])
    data_from_post.append(request.form['commits'])
    data_from_post.append(request.form['pull-requests'])
    data_from_post.append(request.form['contributors'])

    data = [int(x) if x.isdigit() else 0 for x in data_from_post]
    model_data = [data]

    filename = '../training_data/data_github.csv'
    classifier = GitHubClassifier(filename)

    model = request.form['model-choice']
    if (model == 'gradient-boosting'):
        classifier.load_model('../trained_models/gradien_boosting.pkl')
        model = 'Градиентный бустинг'
    if model == 'random-forest':
        classifier.load_model('../trained_models/random_forest.pkl')
        model = 'Случайный лес'
    class_rf = classifier.predict_class(model_data)
    return render_template('result.html', class_popularity=class_rf, model_data=data, repo_name=repo, model_name=model)


if __name__ == '__main__':
    app.run(debug=True)
