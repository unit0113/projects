#pragma once
#include <iostream>
#include <vector>
#include <stdexcept>
using namespace std;


template<typename T>
class LinkedList {
    public:
        class Node {
            friend class LinkedList;
            public:
                Node(T in_data) {data = in_data;};
                ~Node() = default;
                Node* next = nullptr;
                Node* prev = nullptr;
                T data;
        };

        LinkedList() = default;
        LinkedList(const LinkedList& otherList);
        LinkedList& operator=(const LinkedList& otherList);
        ~LinkedList();
        void AddHead(T data);
        void AddTail(T data);
        void AddNodesHead(T* data, int size);
        void AddNodesTail(T* data, int size);
        size_t NodeCount() const {return count;};
        void PrintForward() const;
        void PrintReverse() const;
        Node* Head() const {return head;};
        Node* Tail() const {return tail;};
        Node* GetNode(int index) const;
        Node* Find(T val) const;
        void FindAll(vector<Node*>& pVec, T val) const;

        T operator[](int index) const;

    private:
        Node* head = nullptr;
        Node* tail = nullptr;
        size_t count {};
        void copy(const LinkedList& otherList);
        void reset();
};


template<typename T>
void LinkedList<T>::copy(const LinkedList& otherList) {
    Node* currNode = otherList.tail;
    while (currNode != nullptr) {
        AddHead(currNode->data);
        currNode = currNode->prev;
    }
}

template<typename T>
void LinkedList<T>::reset() {
    count = 0;
    tail = nullptr;
    Node* current = head;
    Node* next_node;
    head = nullptr;
    while (current != nullptr) {
        next_node = current->next;
        delete current;
        current = next_node;
    }
}

template<typename T>
LinkedList<T>::LinkedList(const LinkedList& otherList) {
    copy(otherList);
}

template<typename T>
LinkedList<T>& LinkedList<T>::operator=(const LinkedList<T>& otherList) {
    reset();
    copy(otherList);
    return *this;
}

template<typename T>
LinkedList<T>::~LinkedList() {
    reset();
}


template<typename T>
void LinkedList<T>::AddHead(T data) {
    Node* newNode = new Node(data);
    if (count != 0) {
        Node* currHead = head;
        newNode->next = currHead;
        currHead->prev = newNode;
    } else {
        // Also set tail to new node if list is empty
        tail = newNode;
    }
    head = newNode;
    ++count;
}

template<typename T>
void LinkedList<T>::AddTail(T data) {
    Node* newNode = new Node(data);
    if (count != 0) {
        Node* currTail = tail;
        newNode->prev = currTail;
        currTail->next = newNode;
    } else {
        // Also set head to new node if list is empty
        head = newNode;
    }
    tail = newNode;
    ++count;
}

template<typename T>
void LinkedList<T>::AddNodesHead(T* data, int size) {
    // Add items in reverse order to maintain original order
    for (int i = size - 1; i >= 0; --i) {
        AddHead(data[i]);
    }
}

template<typename T>
void LinkedList<T>::AddNodesTail(T* data, int size) {
    // Add items in order to maintain original order
    for (int i {}; i < size; ++i) {
        AddTail(data[i]);
    }
}

template<typename T>
void LinkedList<T>::PrintForward() const {
    Node* currNode = head;
    while (currNode != nullptr) {
        cout << currNode->data << endl;
        currNode = currNode->next;
    }
}

template<typename T>
void LinkedList<T>::PrintReverse() const {
    Node* currNode = tail;
    while (currNode != nullptr) {
        cout << currNode->data << endl;
        currNode = currNode->prev;
    }
}

template<typename T>
typename LinkedList<T>::Node* LinkedList<T>::GetNode(int index) const {
    if (index >= static_cast<int>(count)) {
        throw out_of_range("Index out of range");
    }
    Node* currNode = head;
    for (int i{}; i < index; ++i) {
        currNode = currNode->next;
    }
    return currNode;
}

template<typename T>
typename LinkedList<T>::Node* LinkedList<T>::Find(T val) const {
    Node* currNode = head;
    for (size_t i{}; i < count; ++i) {
        if (val == currNode->data) {
            return currNode;
        }
    }
    return nullptr;
}

template<typename T>
void LinkedList<T>::FindAll(vector<Node*>& pVec, T val) const {
    Node* currNode = head;
    while (currNode != nullptr) {
        if (currNode->data == val) {
            pVec.push_back(currNode);
        }
        currNode = currNode->next;
    }
}

template<typename T>
T LinkedList<T>::operator[](int index) const {
    return GetNode(index)->data;
}