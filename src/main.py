from flask import Flask, render_template, request, redirect, url_for

from GHClassifier import GHClassifier
from GHRegressor import GHRegressor

app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/redirect', methods=['POST'])
def redirect_page():
    option = request.form['model-option']
    if option == 'classification':
        return redirect(url_for('classifier'))
    elif option == 'regression':
        return redirect(url_for('regressor'))


@app.route('/classifier')
def classifier():
    return render_template('classifier.html')


@app.route('/regressor')
def regressor():
    return render_template('regressor.html')


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
    source = request.args.get('source')
    return render_template('result.html', popularity=popularity, repo=repo, forks=forks, stars=stars, issues=issues,
                           commits=commits, requests=requests, contributors=contributors, repo_name=repo, model_name=model, source=source)


@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        source = request.form['source']
        repo = request.form['repo-name']
        model = request.form['model-choice']
        forks = int(request.form['forks'])
        issues = int(request.form['issues'])
        commits = int(request.form['commits'])
        requests = int(request.form['pull-requests'])
        contributors = int(request.form['contributors'])
        filename = '../training_data/data_github.csv'

        if source == 'classifier':
            stars = int(request.form['stars'])
            model_name = {
                'gradient-boosting': 'Градиентный бустинг',
                'random-forest': 'Случайный лес',
                'decision-tree': 'Деревья решений'
            }.get(model)

            classifier = GHClassifier(filename, 0.2, 10)
            if model in ('gradient-boosting', 'random-forest', 'decision-tree'):
                getattr(classifier, f'train_{model.replace("-", "_")}')(
                    5 if model != 'decision-tree' else None)
            popularity = classifier.predict(
                [[forks, stars, issues, commits, requests, contributors]])
        elif source == 'regressor':
            model_name = {
                'gradient-boosting': 'Градиентный бустинг',
                'random-forest': 'Случайный лес'
            }.get(model)

            regressor = GHRegressor(filename, 0.2, 10)
            if model in ('gradient-boosting', 'random-forest'):
                getattr(regressor, f'train_{model.replace("-", "_")}')(
                    50 if model == 'gradient-boosting' else 10)
            popularity = regressor.predict(
                [[forks, issues, commits, requests, contributors]])

        return redirect(url_for('result', popularity=popularity, repo=repo, forks=forks, stars=request.form.get('stars', 0), issues=issues,
                                commits=commits, requests=requests, contributors=contributors, repo_name=repo, model=model_name, source=source))


if __name__ == '__main__':
    app.run(debug=True)
