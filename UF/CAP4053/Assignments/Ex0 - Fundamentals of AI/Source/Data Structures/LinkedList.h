#pragma once
// LinkedList class should go in the "ufl_cap4053::fundamentals" namespace!
namespace ufl_cap4053 { namespace fundamentals {
	template <typename T>
	class Node;
	template <typename T>
	class Iterator;
	template <typename T>
	class LinkedList;

} }  // namespace ufl_cap4053::fundamentals

using ufl_cap4053::fundamentals::LinkedList;
using ufl_cap4053::fundamentals::Node;
using ufl_cap4053::fundamentals::Iterator;

template <typename T>
class Node {
private:
	T data;
	Node* next{ nullptr };
	Node* prev{ nullptr };

public:
	Node(T val) : data(val) {};
	Node(T val, Node* node) : data(val), prev(node) {};
	~Node() = default;
	T getData() { return data; };
	Node* getNext() const { return next; };
	void setNext(Node* node) { next = node; };
	Node* getPrev() const { return prev; };
	void setPrev(Node* node) { prev = node; };
};

template <typename T>
class Iterator {
private:
	Node<T>* current{ nullptr };

public:
	Iterator() = default;
	Iterator(Node<T>* node) : current(node) {};
	T operator*() { return current->getData(); };
	Iterator& operator++();
	bool operator==(const Iterator& rhs) { return current == rhs.current; };
	bool operator!=(const Iterator& rhs) { return current != rhs.current; };
};

template <typename T>
class LinkedList {
	public:
		LinkedList() = default;
		~LinkedList();

		Iterator<T> begin() const { return Iterator<T>(head); };
		Iterator<T> end() const { return Iterator<T>(); };
		bool isEmpty() const { return !head; };
		T getFront() const;
		T getBack() const;
		void enqueue(T data);
		void dequeue() { deleteNode(head); };
		void pop() { deleteNode(tail); };
		void clear();
		bool contains(T data) const;
		void remove(T data);


	private:
		Node<T>* head{ nullptr };
		Node<T>* tail{ nullptr };
		void deleteNode(Node<T>* node);
		Node<T>* findNode(T data) const;
};

template<typename T>
Iterator<T>& Iterator<T>::operator++() {
	current = current->getNext();
	return *this;
}

template<typename T>
LinkedList<T>::~LinkedList() {
	clear();
}

template<typename T>
void LinkedList<T>::clear() {
	tail = nullptr;
	Node<T>* current = head;
	Node<T>* next_node;
	head = nullptr;
	while (current) {
		next_node = current->getNext();
		delete current;
		current = next_node;
	}
}

template<typename T>
T LinkedList<T>::getFront() const {
	return head->getData();
}

template<typename T>
T LinkedList<T>::getBack() const {
	return tail->getData();
}

template<typename T>
void LinkedList<T>::enqueue(T data) {
	Node<T>* newNode = new Node<T>(data);
	// Adjust head if empty
	if (isEmpty())
		head = newNode;
	else {
		newNode->setPrev(tail);
		tail->setNext(newNode);
	}
	tail = newNode;
}

template<typename T>
void LinkedList<T>::deleteNode(Node<T>* node) {
	// Return if nullptr
	if (!node) { return; }

	// Attach prev to next
	if (node->getPrev())
		node->getPrev()->setNext(node->getNext());
	else
		head = node->getNext();

	// Attach next to prev
	if (node->getNext())
		node->getNext()->setPrev(node->getPrev());
	else
		tail = node->getPrev();

	delete node;
}

template<typename T>
Node<T>* LinkedList<T>::findNode(T data) const {
	Node<T>* current = head;
	while (current) {
		if (current->getData() == data)
			return current;
		current = current->getNext();
	}
	return current;
}

template<typename T>
bool LinkedList<T>::contains(T data) const {
	if (findNode(data)) { return true; }
	return false;
}

template<typename T>
void LinkedList<T>::remove(T data) {
	deleteNode(findNode(data));
}