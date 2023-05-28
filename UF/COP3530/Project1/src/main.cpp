#include <iostream>
#include <string>
#include <deque>

using namespace std;

/* Note: 
	1. You will have to comment main() when unit testing your code because catch uses its own main().
	2. You will submit this main.cpp file and any header files you have on Gradescope. 
*/

struct Node {
	string name;
	unsigned int ID;
	unsigned int height = 1;
	Node* parent = nullptr;
	Node* l_child = nullptr;
	Node* r_child = nullptr;

	Node(string new_name, unsigned int new_ID, Node* new_parent=nullptr) : name(new_name), ID(new_ID), parent(new_parent) {}
};


class AVL_Tree {
	private:
		Node* head = nullptr;

		// Helper methods
		Node* insert_helper(string name, unsigned int ID, Node* root);
		unsigned int get_height(Node* root);
		Node* rebalance(Node* root);
		unsigned int get_balance_factor(Node* root);
		Node* left_rotation(Node* old_root);
		Node* right_rotation(Node* old_root);
		void in_order_helper(Node* root, deque<Node*>& nodes) const;
		void print_helper(deque<Node*>& nodes) const;
		void pre_order_helper(Node* root, deque<Node*>& nodes) const;
		void post_order_helper(Node* root, deque<Node*>& nodes) const;

	public:
		void insert(string name, unsigned int ID);
		void remove(const unsigned int& remove_ID);
		void search(const unsigned int& search_ID) const;
		void search(const string& search_name) const;
		void printInOrder() const;
		void printPreOrder() const;
		void printPostOrder() const;
		void printLevelCount() const;
		void removeInOrder(unsigned int N);
};


void AVL_Tree::insert(string name, unsigned int ID) {
	// For empty tree
	if (head == nullptr) {
		head = new Node(name, ID);
		cout << "successful" << endl;
		return;
	}

	// Else
	insert_helper(name, ID, head);
}


Node* AVL_Tree::insert_helper(string name, unsigned int ID, Node* root) {
	// Base case for reaching end of tree
	if (root == nullptr) {
		cout << "successful" << endl;
		return new Node(name, ID);
	}
	// Go to left
	if (ID < root->ID)  {
		Node* left_sub_root = insert_helper(name, ID, root->l_child);
		root->l_child = left_sub_root;
		left_sub_root->parent = root;
	}

	// If ID already in tree
	else if (ID == root->ID) {
		cout << "unsuccessful" << endl;
		return root;
	}

	// Go right
	else {
		Node* right_sub_root = insert_helper(name, ID, root->r_child);
		root->r_child = right_sub_root;
		right_sub_root->parent = root;
	}

	// Update root's height
	root->height = 1 + max(get_height(root->l_child), get_height(root->r_child));

	// Rebalance as required
	return rebalance(root);
}


unsigned int AVL_Tree::get_height(Node* root) {
	if (root == nullptr) {return 0;}
	return root->height;
}


Node* AVL_Tree::rebalance(Node* root) {
	int balance_factor = get_balance_factor(root);

	// Right or left-right rotation
	if (balance_factor > 1) {
		if (get_balance_factor(root->l_child) < 0) {
			root->l_child = left_rotation(root->l_child);
		}
		return right_rotation(root);
	}
	
	// Left or right-left rotation
	else if (balance_factor < -1) {
		if (get_balance_factor(root->r_child) > 0) {
			root->r_child = right_rotation(root->r_child);
		}
		return left_rotation(root);
	}

	else {return root;}
}


unsigned int AVL_Tree::get_balance_factor(Node* root) {
	if (root == nullptr) {return 0;}
	return get_height(root->l_child) - get_height(root->r_child);
}


Node* AVL_Tree::left_rotation(Node* old_root) {
	Node* new_root = old_root->r_child;
	old_root->r_child = new_root->l_child;

	// Assign parent to left child of new root
	if (new_root->l_child) {
		new_root->l_child->parent = old_root;
	}

	// Adjust parent/child relationship with new root and it's new parent
	if (old_root->parent) {
		// Check which child the old root was
		if (old_root->ID == old_root->parent->l_child->ID) {
			old_root->parent->l_child = new_root;
		} else {
			old_root->parent->r_child = new_root;
		}
	} 
	
	// If old root was the head
	else {
		head = new_root;
	}

	// Clean up final connection and update heights
	new_root->l_child = old_root;
	new_root->parent = old_root->parent;
	old_root->parent = new_root;
	old_root->height = 1 + max(get_height(old_root->l_child), get_height(old_root->r_child));
	new_root->height = 1 + max(get_height(new_root->l_child), get_height(new_root->r_child));

	return new_root;
}


Node* AVL_Tree::right_rotation(Node* old_root) {
	Node* new_root = old_root->l_child;
	old_root->l_child = new_root->r_child;

	// Assign parent to left child of new root
	if (new_root->r_child) {
		new_root->r_child->parent = old_root;
	}

	// Adjust parent/child relationship with new root and it's new parent
	if (old_root->parent) {
		// Check which child the old root was
		if (old_root->ID == old_root->parent->l_child->ID) {
			old_root->parent->l_child = new_root;
		} else {
			old_root->parent->r_child = new_root;
		}
	} 
	
	// If old root was the head
	else {
		head = new_root;
	}

	// Clean up final connection and update heights
	new_root->r_child = old_root;
	new_root->parent = old_root->parent;
	old_root->parent = new_root;
	old_root->height = 1 + max(get_height(old_root->l_child), get_height(old_root->r_child));
	new_root->height = 1 + max(get_height(new_root->l_child), get_height(new_root->r_child));

	return new_root;
}


void AVL_Tree::remove(const unsigned int& remove_ID) {

}


void AVL_Tree::search(const unsigned int& search_ID) const {
	Node* curr = head;
	while (curr) {
		if (curr->ID == search_ID) {
			cout << curr->name << endl;
			return;
		}
		curr = (search_ID < curr->ID) ? curr->l_child : curr->r_child;
	}
	cout << "unsuccessful" << endl;
}


void AVL_Tree::search(const string& search_name) const {
	deque<Node*> nodes;
	pre_order_helper(head, nodes);
	for (Node* node: nodes) {
		if (node->name == search_name) {
			cout << node->ID << endl;
		}
	}
}


void AVL_Tree::printInOrder() const {
	deque<Node*> nodes;
	in_order_helper(head, nodes);
	print_helper(nodes);
}


void AVL_Tree::in_order_helper(Node* root, deque<Node*>& nodes) const {
	if (root) {
		in_order_helper(root->l_child, nodes);
		nodes.push_back(root);
		in_order_helper(root->r_child, nodes);
	}
}


void AVL_Tree::printPreOrder() const {
	deque<Node*> nodes;
	pre_order_helper(head, nodes);
	print_helper(nodes);
}


void AVL_Tree::pre_order_helper(Node* root, deque<Node*>& nodes) const {
	if (root) {
		nodes.push_back(root);
		in_order_helper(root->l_child, nodes);
		in_order_helper(root->r_child, nodes);
	}
}

		
void AVL_Tree::printPostOrder() const {
	deque<Node*> nodes;
	post_order_helper(head, nodes);
	print_helper(nodes);
}


void AVL_Tree::post_order_helper(Node* root, deque<Node*>& nodes) const {
	if (root) {
		in_order_helper(root->l_child, nodes);
		in_order_helper(root->r_child, nodes);
		nodes.push_back(root);
	}
}


void AVL_Tree::print_helper(deque<Node*>& nodes) const {
	cout << nodes.front()->name;
	nodes.pop_front();
	while (!nodes.empty()) {
		cout << ", " << nodes.front()->name;
		nodes.pop_front();
	}
	cout << endl;
}


void AVL_Tree::printLevelCount() const {
	if (!head) {
		cout << 0 << endl;
	}
	cout << head->height << endl;
}


void AVL_Tree::removeInOrder(unsigned int N) {

}






















int main(){
	AVL_Tree tree;
	tree.insert("Bob", 42);
	tree.insert("Alice", 25);
	tree.insert("Charlie", 72);
	tree.insert("Alice", 94);
	tree.insert("Bad", 25);
	tree.printInOrder();
	tree.printPreOrder();
	tree.printPostOrder();
	tree.printLevelCount();
	tree.search("Alice");
}

