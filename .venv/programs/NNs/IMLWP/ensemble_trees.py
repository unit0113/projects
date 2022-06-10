from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=0)

forest = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)
print(f'Training Accuracy: {forest.score(X_train, y_train):.3f}')
print(f'Test Accuracy: {forest.score(X_test, y_test):.3f}')

gbrt = GradientBoostingClassifier(random_state=0, max_depth=1).fit(X_train, y_train)
print(f'Training Accuracy: {gbrt.score(X_train, y_train):.3f}')
print(f'Test Accuracy: {gbrt.score(X_test, y_test):.3f}')
