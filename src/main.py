from flask import Flask, render_template, request, redirect, url_for

from GHClassifier import GHClassifier
from GHRegressor import GHRegressor

app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result')
def result():
    model = request.args.get('model')
    popularity = request.args.get('popularity')

    repo = request.args.get('repo')
    forks = request.args.get('forks')
    stars = request.args.get('stars')
    issues = request.args.get('issues')
    commits = request.args.get('commits')
    requests = request.args.get('requests')
    contributors = request.args.get('contributors')
    return render_template('result.html', class_popularity=popularity, repo=repo, forks=forks, stars=stars, issues=issues,
                           commits=commits, requests=requests, contributors=contributors, repo_name=repo, model_name=model)


@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        repo = request.form['repo-name']
        forks = int(request.form['forks'])
        stars = int(request.form['stars'])
        issues = int(request.form['issues'])
        commits = int(request.form['commits'])
        requests = int(request.form['pull-requests'])
        contributors = int(request.form['contributors'])

        model = request.form['model-choice']

        filename = '../training_data/data_github.csv'
        classifier = GHClassifier(filename, 0.2, 10)
        if model == 'gradient-boosting':
            classifier.train_gradient_boosting(10, 42, 1)
            model = 'Градиентный бустинг'
        if model == 'random-forest':
            classifier.train_random_forest(10, 42)
            model = 'Случайный лес'
        if model == 'decision-tree':
            classifier.train_decision_tree(42)
            model = 'Деревья решений'
        if model == 'adaboost':
            classifier.train_adaboost(10, 42)
            model = 'AdaBoost'
        if model == 'naive-bayes':
            classifier.train_naive_bayes()
            model = 'Наивный байесовский классификатор'
        print([forks, stars, issues, commits, requests, contributors])
        class_rf = classifier.predict([
            [forks, stars, issues, commits, requests, contributors]])
        return redirect(url_for('result', popularity=class_rf, repo=repo, forks=forks, stars=stars, issues=issues,
                                commits=commits, requests=requests, contributors=contributors, repo_name=repo, model=model))


if __name__ == '__main__':
    app.run(debug=True)
