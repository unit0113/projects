from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


cancer = load_breast_cancer()

# Standard practice
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=0)
scaler = StandardScaler()
X_train_scaled = StandardScaler.fit_transform(X_train)
X_test_scaled = StandardScaler.transform(X_test)

# Skipping for demo
X_scaled = scaler.fit_transform(cancer.data)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
