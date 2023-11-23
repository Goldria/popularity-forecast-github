import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


file_path = 'data_github.csv'
data = pd.read_csv(file_path)
num_classes = 3  # Количество классов -- поменять

data['popularity'] = (
    data['stars'] / data['stars'].max() +
    0.7 * data['issues'] / data['issues'].max() +
    0.5 * data['commits'] / data['commits'].max() +
    0.3 * data['forks'] / data['forks'].max() +
    0.3 * data['pull_requests'] / data['pull_requests'].max()
) * 10

data['popularity_class'] = pd.cut(
    data['popularity'], bins=num_classes, labels=['Low', 'Medium', 'High'])

X = data.drop(['repo_name', 'popularity', 'popularity_class'], axis=1)
y = data['popularity_class']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

new_data = [[32773, 299599, 1131, 81, 3955]]
new_data_scaled = scaler.transform(new_data)

predicted_popularity = model.predict(new_data_scaled)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')

print(f'Предсказанный уровень популярности проекта: {predicted_popularity}')
print(f'Accuracy: {accuracy}')
print(f'F1-score: {f1}')

gb_classifier = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.2, random_state=42)

gb_classifier.fit(X_train, y_train)

y_pred = gb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

predicted_class = gb_classifier.predict(new_data)

print(f'Прогнозируемый класс: {predicted_class}')
print(f'Accuracy: {accuracy}')
print(f'F1-score: {f1}')
