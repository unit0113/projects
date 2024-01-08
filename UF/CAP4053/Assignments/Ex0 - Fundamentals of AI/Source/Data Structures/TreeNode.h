#pragma once

#include <vector>
#include <queue>

// TreeNode class should go in the "ufl_cap4053::fundamentals" namespace!
namespace ufl_cap4053 { namespace fundamentals {
	template <typename T>
	class TreeNode;
}}  // namespace ufl_cap4053::fundamentals

using ufl_cap4053::fundamentals::TreeNode;

template <typename T>
class TreeNode {
private:
	T data;
	std::vector<TreeNode<T>*> children;

public:
	TreeNode<T>() = default;
	TreeNode<T>(T element) : data(element) {};
	const T& getData() const { return data; };
	size_t getChildCount() const { return children.size(); };
	TreeNode<T>* getChild(size_t index) { return children[index]; };
	TreeNode<T>* getChild(size_t index) const { return children[index]; };
	void addChild(TreeNode<T>* child) { children.push_back(child); };
	TreeNode<T>* removeChild(size_t index);
	void breadthFirstTraverse(void (*dataFunction)(const T)) const;
	void preOrderTraverse(void (*dataFunction)(const T)) const;
	void postOrderTraverse(void (*dataFunction)(const T)) const;
};

template<typename T>
TreeNode<T>* TreeNode<T>::removeChild(size_t index) {
	TreeNode<T>* return_node = getChild(index);
	children.erase(children.begin() + index);
	return return_node;
}

template<typename T>
void TreeNode<T>::breadthFirstTraverse(void (*dataFunction)(const T)) const {
	dataFunction(getData());

	// Initialize and load q
	std::queue<TreeNode<T>*> q;
	for (TreeNode<T>* child : children) {
		q.push(child);
	}

	// Level order
	while (!q.empty()) {
		TreeNode<T>* node = q.front();
		q.pop();
		dataFunction(node->getData());
		for (size_t i{}; i < node->getChildCount(); ++i) {
			q.push(node->getChild(i));
		}
	}
}

template<typename T>
void TreeNode<T>::preOrderTraverse(void (*dataFunction)(const T)) const {
	dataFunction(getData());
	for (TreeNode<T>* child : children) {
		child->preOrderTraverse(dataFunction);
	}
}

template<typename T>
void TreeNode<T>::postOrderTraverse(void (*dataFunction)(const T)) const {
	for (TreeNode<T>* child : children) {
		child->postOrderTraverse(dataFunction);
	}
	dataFunction(getData());
}