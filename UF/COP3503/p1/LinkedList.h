#pragma once
#include <iostream>
#include <vector>
#include <stdexcept>
using namespace std;


template<typename T>
class LinkedList {
    public:
        // Nested Node class
        class Node {
            friend class LinkedList;
            public:
                Node(T in_data) {data = in_data;};
                ~Node() = default;
                Node* next = nullptr;
                Node* prev = nullptr;
                T data;
        };

        LinkedList() = default;                                 // Default constructor
        LinkedList(const LinkedList& otherList);                // Copy constructor
        LinkedList& operator=(const LinkedList& otherList);     // Copy assignement
        ~LinkedList();                                          // Destructor
        
        void Clear();
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
        void InsertAfter(Node* node, T data);
        void InsertBefore(Node* node, T data);
        void InsertAt(T data, int index);
        bool RemoveHead();
        bool RemoveTail();
        bool RemoveAt(int index);
        int Remove(T val);
        void PrintForwardRecursive(Node* node) const;
        void PrintReverseRecursive(Node* node) const;

        T operator[](int index) const;
        bool operator==(const LinkedList& otherList);

    private:
        Node* head = nullptr;
        Node* tail = nullptr;
        size_t count {};
        void copy(const LinkedList& otherList);
        void deleteNode(Node* node);
};

//********************* Function Definitions ****************************

// Helper function for copy construction and copy assignment
template<typename T>
void LinkedList<T>::copy(const LinkedList& otherList) {
    Node* currNode = otherList.tail;
    while (currNode != nullptr) {
        AddHead(currNode->data);
        currNode = currNode->prev;
    }
}

// Helper function for destructor and copy assignment
template<typename T>
void LinkedList<T>::Clear() {
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
    Clear();
    copy(otherList);
    return *this;
}

template<typename T>
LinkedList<T>::~LinkedList() {
    Clear();
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
    while (currNode) {
        cout << currNode->data << endl;
        currNode = currNode->next;
    }
}

template<typename T>
void LinkedList<T>::PrintReverse() const {
    Node* currNode = tail;
    while (currNode) {
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
    while (currNode)  {
        if (currNode->data == val) {
            return currNode;
        }
        currNode = currNode->next;
    }

    return currNode;
}

template<typename T>
void LinkedList<T>::FindAll(vector<Node*>& pVec, T val) const {
    Node* currNode = head;
    while (currNode) {
        if (currNode->data == val) {
            pVec.push_back(currNode);
        }
        currNode = currNode->next;
    }
}

template<typename T>
void LinkedList<T>::InsertAfter(Node* node, T data) {
    Node* newNode = new Node(data);
    newNode->next = node->next;
    newNode->prev = node;
    node->next = newNode;

    // Check if new tail
    if (newNode->next) {
        newNode->next->prev = newNode;
    } else {
        tail = newNode;
    }
    ++count;
}

template<typename T>
void LinkedList<T>::InsertBefore(Node* node, T data) {
    Node* newNode = new Node(data);
    newNode->prev = node->prev;
    newNode->next = node;
    node->prev = newNode;

    // Check if new head
    if (newNode->prev) {
        newNode->prev->next = newNode;
    } else {
        head = newNode;
    }
    ++count;
}

template<typename T>
void LinkedList<T>::InsertAt(T data, int index) {
    // If inserting at last index (new tail)
    if (index == static_cast<int>(count)) {
        AddTail(data);
    } else {
        InsertBefore(GetNode(index), data);
    }
}

template<typename T>
bool LinkedList<T>::RemoveHead() {
    if (!head) {
        return false;
    }

    deleteNode(head);
    return true;
}

template<typename T>
bool LinkedList<T>::RemoveTail() {
    if (!tail) {
        return false;
    }

    deleteNode(tail);
    return true;
}

template<typename T>
bool LinkedList<T>::RemoveAt(int index) {
    if (index >= static_cast<int>(count)) {
        return false;
    }

    deleteNode(GetNode(index));
    return true;
}

template<typename T>
int LinkedList<T>::Remove(T val) {
    Node* currNode = head;
    Node* nextNode;
    int delCount {};
    while (currNode) {
        nextNode = currNode->next;
        if (currNode->data == val) {
            deleteNode(currNode);
            ++delCount;
        }
        currNode = nextNode;
    }
    return delCount;
}

// Helper function to delete nodes
template<typename T>
void LinkedList<T>::deleteNode(Node* node) {
    if (node->prev) {
        node->prev->next = node->next;
    } else {
        head = node->next;
    }

    if (node->next) {
        node->next->prev = node->prev;
    } else {
        tail = node->prev;
    }    

    --count;
    delete node;
}

template<typename T>
void LinkedList<T>::PrintForwardRecursive(Node* node) const {
    if (node) {
        cout << node->data << endl;
        PrintForwardRecursive(node->next);
    }
}

template<typename T>
void LinkedList<T>::PrintReverseRecursive(Node* node) const {
    if (node) {
        cout << node->data << endl;
        PrintReverseRecursive(node->prev);
    }
}


//Operators
template<typename T>
T LinkedList<T>::operator[](int index) const {
    return GetNode(index)->data;
}

template<typename T>
bool LinkedList<T>::operator==(const LinkedList& otherList) {
    if (count != otherList.count) {
        return false;
    }

    Node* currNodeThis = head;
    Node* currNodeOther = otherList.head;
    while (currNodeThis) {
        if (currNodeThis->data != currNodeOther->data) {
            return false;
        }
        currNodeThis = currNodeThis->next;
        currNodeOther = currNodeOther->next;
    }
    return true;
}