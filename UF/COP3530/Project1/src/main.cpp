#include <iostream>
#include <memory>
#include <string>

using namespace std;

/* Note: 
	1. You will have to comment main() when unit testing your code because catch uses its own main().
	2. You will submit this main.cpp file and any header files you have on Gradescope. 
*/

struct Node {
	string name;
	unsigned int ID;
	unsigned int height = 1;
	shared_ptr<Node> parent = nullptr;
	shared_ptr<Node> l_child = nullptr;
	shared_ptr<Node> r_child = nullptr;

	Node(string new_name, unsigned int new_ID, shared_ptr<Node> new_parent=nullptr) : name(new_name), ID(new_ID), parent(new_parent) {}
};


class AVL_Tree {
	private:
		shared_ptr<Node> head = nullptr;
		shared_ptr<Node> insert_helper(string name, unsigned int ID, shared_ptr<Node> root);
		unsigned int get_height(shared_ptr<Node> root);
		shared_ptr<Node> rebalance(shared_ptr<Node> root);
		unsigned int get_balance_factor(shared_ptr<Node> root);
		shared_ptr<Node> left_rotation(shared_ptr<Node> old_root);
		shared_ptr<Node> right_rotation(shared_ptr<Node> old_root);

	public:
		void insert(string name, unsigned int ID);
		void remove(const unsigned int& ID);
		void search(const unsigned int& ID) const;
		void search(const string& name) const;
		void printInOrder() const;
		void printPreOrder() const;
		void printPostOrder() const;
		void printLevelCount() const;
		void removeInOrder(unsigned int N);
};


void AVL_Tree::insert(string name, unsigned int ID) {
	// For empty tree
	if (head == nullptr) {
		head = make_shared<Node>(name, ID);
		cout << "successful" << endl;
		return;
	}

	// Else
	insert_helper(name, ID, head);
}


shared_ptr<Node> AVL_Tree::insert_helper(string name, unsigned int ID, shared_ptr<Node> root) {
	// Base case for reaching end of tree
	if (root == nullptr) {
		cout << "successful" << endl;
		return make_shared<Node>(name, ID);
	}
	// Go to left
	if (ID < root->ID)  {
		shared_ptr<Node> left_sub_root = insert_helper(name, ID, root->l_child);
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
		shared_ptr<Node> right_sub_root = insert_helper(name, ID, root->r_child);
		root->r_child = right_sub_root;
		right_sub_root->parent = root;
	}

	// Update root's height
	root->height = 1 + max(get_height(root->l_child), get_height(root->r_child));

	// Rebalance as required
	return rebalance(root);
}


unsigned int AVL_Tree::get_height(shared_ptr<Node> root) {
	if (root == nullptr) {return 0;}
	return root->height;
}


shared_ptr<Node> AVL_Tree::rebalance(shared_ptr<Node> root) {
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


unsigned int AVL_Tree::get_balance_factor(shared_ptr<Node> root) {
	if (root == nullptr) {return 0;}
	return get_height(root->l_child) - get_height(root->r_child);
}


shared_ptr<Node> AVL_Tree::left_rotation(shared_ptr<Node> old_root) {
	shared_ptr<Node> new_root = old_root->r_child;
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


shared_ptr<Node> AVL_Tree::right_rotation(shared_ptr<Node> old_root) {
	shared_ptr<Node> new_root = old_root->l_child;
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






int main(){
	AVL_Tree tree;
	tree.insert("Bob", 42);
}

