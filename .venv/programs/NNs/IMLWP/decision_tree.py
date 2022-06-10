from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import graphviz


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_train, y_train)
print(f'Training Accuracy: {tree.score(X_train, y_train):.3f}')
print(f'Test Accuracy: {tree.score(X_test, y_test):.3f}')

export_graphviz(tree, out_file=r'NNs\IMLWP\tree.dot', class_names=['Malignant', 'Benign'],
                feature_names=cancer.feature_names, impurity=False, filled=True)

with open(r'NNs\IMLWP\tree.dot') as file:
    dot_graph = file.read()
    graph = graphviz.Source(dot_graph)
    graph.view()
