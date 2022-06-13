from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


iris = load_iris()
logreg = LogisticRegression()

scores = cross_val_score(logreg, iris.data, iris.target, cv=5, n_jobs=4)
round_scores = [round(score, 3) for score in scores]
print(f'Cross Validation Scores: {round_scores}')
print(f'Average Cross Validation Score: {scores.mean():.3f}')

# Part 2
X_train_val, X_test, y_train_val, y_test = train_test_split(iris.data, iris.target, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, random_state=1)

values = [0.001, 0.01, 0.1, 1, 10, 100]
best_score = 0
for gamma in values:
    for C in values:
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_val, y_val)
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}

svm = SVC(**best_parameters)
svm.fit(X_train_val, y_train_val)
test_score = svm.score(X_test, y_test)
print(f'Best val score: {best_score:.3f}')
print('Best Parameters: ', best_parameters)
print(f'Test score: {test_score:.3f}')
