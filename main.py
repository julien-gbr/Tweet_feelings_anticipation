# Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

file_path = r"C:\Users\julga\Documents\tweet_dataset\training.1600000.processed.noemoticon.csv"

df = pd.read_csv(file_path, encoding='latin1', header=None, names=['feeling', 'id', 'date', 'query', 'user', 'text'])

X = df['text']
y = df['feeling']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('logreg', LogisticRegression(solver='liblinear'))
])

param_grid = {
    'tfidf__max_df': [0.9, 0.95, 1.0],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'logreg__C': [0.1, 1, 10, 100],
    'logreg__penalty': ['l1', 'l2']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=3)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, predictions))

# Best parameters: {'logreg__C': 1, 'logreg__penalty': 'l2', 'tfidf__max_df': 0.9, 'tfidf__ngram_range': (1, 2)}
# Accuracy: 82.42%
