from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=0)

svc = SVC().fit(X_train, y_train)
print(f'Training Accuracy: {svc.score(X_train, y_train):.3f}')
print(f'Test Accuracy: {svc.score(X_test, y_test):.3f}')

# Normalize data for SVC (seems to do it automatically), but supposed to prevent overfitting
min_on_train = X_train.min(axis=0)
range_on_train = (X_train - min_on_train).max(axis=0)
X_train_scaled = (X_train - min_on_train) / range_on_train
X_test_scaled = (X_test - min_on_train) / range_on_train

# Fit with normalized data
svc = SVC().fit(X_train_scaled, y_train)
print(f'Training Accuracy: {svc.score(X_train_scaled, y_train):.3f}')
print(f'Test Accuracy: {svc.score(X_test_scaled, y_test):.3f}')