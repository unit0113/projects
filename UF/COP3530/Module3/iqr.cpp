class Node {
    public:
        int value;
        Node* next = nullptr;
};


//https://stackoverflow.com/questions/73735926/how-to-find-quartile-3-of-a-linked-list-given-that-you-can-only-iterate-through
float interQuartile(Node* head)
{
    //your code here
    Node* q1_node = head;
    Node* q3_node = head->next;
    Node* end_node = q3_node->next;
    long long count {2};

    while (end_node) {
        if (count % 4 == 1) {q1_node = q1_node->next;}
        if (count % 4 != 3) {q3_node = q3_node->next;}
        end_node = end_node->next;
        ++count;
    }
    
    double q1 = (count % 4 < 2) ? (q1_node->value + q1_node->next->value) / 2.0f : q1_node->value;
    double q3 = (count % 4 < 2) ? (q3_node->value + q3_node->next->value) / 2.0f : q3_node->value;
    return q3 - q1;
   
}