#include <iostream>
#include <vector>
#include <sstream>
using namespace std;


class Node {
    public:
        int data;
        Node* next;
        Node(int val) {data = val; next = nullptr;}
        Node(int val, Node* node) {data = val; next = node;}
};


class Stack_LL {
    private:
        // topPtr points to the top element of the stack
        Node* topPtr;
    public:
        Stack_LL();
        ~Stack_LL();

        bool isEmpty() const;
        void push(int newItem);
        void pop();
        int peek() const;
};

Stack_LL::Stack_LL() {
    topPtr = nullptr;
}

Stack_LL::~Stack_LL() {
    Node* current = topPtr;
    Node* next;
    while (current != nullptr) {
        next = current->next;
        delete current;
        current = next;
    }
}

bool Stack_LL::isEmpty() const {
    return topPtr == nullptr;
}

void Stack_LL::push(int newItem) {
    Node* newNode = new Node(newItem, topPtr);
    topPtr = newNode;

}

void Stack_LL::pop() {
    if (topPtr == nullptr) {return;}
    
    Node* top = topPtr;
    topPtr = topPtr->next;
    delete top;
}

int Stack_LL::peek() const {
    return topPtr->data;
}