class Node {
    public:
        int value;
        Node* next = nullptr;
};


float interQuartile(Node* head)
{
    //your code here
    Node* slow = head;
    Node* fast = head->next->next;
    Node* fastest = fast->next;

    while (fastest != nullptr
           && fastest->next != nullptr
           && fastest->next->next != nullptr
           && fastest->next->next->next != nullptr) {
        slow = slow->next;
        fast = fast->next->next->next;
        fastest = fastest->next->next->next->next;
    }
    
    if (fastest == nullptr ||fastest->next == nullptr) {
        return fast->value - slow->value;
    } else if (fastest->next->next == nullptr) {
        return (fast->next->value + fast->next->next->value) / 2.0f - (slow->value + slow->next->value) / 2.0f;
    } else if (fastest->next->next->next == nullptr) {
        return fast->next->next->value - slow->next->value;
    }
   
}

//https://stackoverflow.com/questions/73735926/how-to-find-quartile-3-of-a-linked-list-given-that-you-can-only-iterate-through
float interQuartile(Node* head)
{
    //your code here
    Node* q1_node = head;
    Node* q3_node = head->next;
    Node* end_node = q3_node->next;
    int count {2};

    while (end_node) {
        if (count % 4 == 1) {q1_node = q1_node->next;}
        if (count % 4 != 3) {q3_node = q3_node->next;}
        end_node = end_node->next;
        ++count;
    }
    
    float q1 = (count % 4 < 2) ? (q1_node->value + q1_node->next->value) / 2.0f : q1_node->value;
    float q3 = (count % 4 < 2) ? (q3_node->value + q3_node->next->value) / 2.0f : q3_node->value;
    return q3 - q1;
   
}