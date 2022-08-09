#include <iostream>
#include <exception>


typedef size_t payload_type;
typedef struct Node Node;


struct Node {
    payload_type payload;
    Node *prev = NULL;
    Node *next = NULL;
};


class Deque {
    private:
        Node *head = NULL;
        Node *tail = NULL;
        size_t length = 0;

    public:
        // Default Constructor
        Deque() {
            
        }

        // Copy Constructor
        Deque(const Deque &other_dec) {
            Node *current = other_dec.head;
            while (current != NULL) {
                Node *new_node = new Node();
                add_back(current->payload);
                current = current->next;
            }
        }

        // Destructor
        ~Deque() {
            tail = NULL;
            Node* current = head;
            Node* next_node;
            head = NULL;
            while (current != NULL) {
                next_node = current->next;
                free(current);
                current = next_node;
            }
        }

        void add_front(payload_type num) {
            Node *new_node = new Node();
            new_node->payload = num;
            new_node->next = head;
            if (head != NULL) {
                head->prev = new_node;
            }
            head = new_node;
            if (tail == NULL) {
                tail = head;
            }
            length++;
        }

        void add_back(payload_type num) {
            Node *new_node = new Node();
            new_node->payload = num;
            new_node->prev = tail;
            if (tail != NULL) {
                tail->next = new_node;
            }
            tail = new_node;
            if (head == NULL) {
                head = tail;
            }
            length++;
        }

        payload_type peek_front() {
            if (head != NULL) {
                return head->payload;
            }
            throw std::out_of_range("Deque is empty");
        }

        payload_type peek_rear() {
            if (tail != NULL) {
                return tail->payload;
            }
            throw std::out_of_range("Deque is empty");
        }

        payload_type pop_front() {
            if (head == NULL) {
                throw std::out_of_range("Deque is empty");
            }

            payload_type num = head->payload;
            Node *temp_node = head;
            head = head->next;
            head->prev = NULL;
            length--;
            if (length <= 1) {
                tail = head;
            }
            free(temp_node);
            return num;
        }

        payload_type pop_back() {
            if (tail == NULL) {
                throw std::out_of_range("Deque is empty");
            }

            payload_type num = tail->payload;
            Node *temp_node = tail;
            tail = tail->prev;
            tail->next = NULL;
            length--;
            if (length <= 1) {
                head = tail;
            }
            free(temp_node);
            return num;
        }

        bool is_empty() {
            return length != 0;
        }

        void print() {
            Node* current_node = head;
            std::cout << "Deque Contents:";
            while (current_node != NULL) {
                std::cout << ' ' << current_node->payload;
                current_node = current_node->next;
            }
            std::cout << std::endl;
        }

        void bubble_sort() {
            if (length < 2) {
                return;
            }

            Node *current = head;
            payload_type temp;
            for (size_t i = 0; i < length; i++) {
                for (size_t j = 0; j < length - i - 1; j++) {
                    if (current->payload > current->next->payload) {
                    temp = current->next->payload;
                    current->next->payload = current->payload;
                    current->payload = temp;
                }
                current = current->next;
                }
                current = head;
            }
        }

        Deque operator+(const Deque &other_deq) const {
            Deque new_deq;
            Node *current = head;
            while (current != NULL) {
                Node *new_node = new Node();
                new_deq.add_back(current->payload);
                current = current->next;
            }

            current = other_deq.head;
            while (current != NULL) {
                Node *new_node = new Node();
                new_deq.add_back(current->payload);
                current = current->next;
            }

            return new_deq;
        }
};


int main() {

    Deque deque = Deque();
    deque.add_front(5);
    deque.add_back(9);
    deque.add_front(2);
    deque.print();
    std::cout << "Popping " << deque.pop_front() << std::endl;
    deque.print();
    deque.add_back(13);
    deque.print();
    std::cout << "Popping " << deque.pop_back() << std::endl;
    deque.print();

    Deque deque2 = Deque();
    deque2.add_front(7);
    deque2.add_front(4);
    Deque deque3 = deque + deque2;
    deque3.print();
    deque3.bubble_sort();
    deque3.print();

    Deque deque4 = deque;
    deque4.print();

    return 0;
}
