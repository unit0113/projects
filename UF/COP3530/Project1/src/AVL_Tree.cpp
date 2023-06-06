#include "AVL_Tree.h"
using namespace std;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Node methods~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
AVL_Tree::Node& AVL_Tree::Node::operator=(const Node& other) {
	/*
	* Copy assignment operator for swapping two nodes
	*/
	ID = other.ID;
	name = other.name;
	return *this;
}



bool AVL_Tree::Node::is_left_child() const {
	/*
	*	Checks if node is the left child of its parent node
	*/
	return parent->l_child && parent->l_child->ID == ID;
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Destructor~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
AVL_Tree::~AVL_Tree() {
	vector<Node*> nodes;
	in_order_helper(head, nodes);
	for (Node* node: nodes) {
		delete node;
	}
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Insertion methods~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
void AVL_Tree::insert(const std::string& name, const std::string& ID) {
	/*
	*	Inserts new item into tree
	*/

	// For empty tree
	if (head == nullptr) {
		head = new Node(name, ID);
		#if (DEBUG != 1)
		cout << "successful" << endl;
		#endif
		return;
	}

	// Else find proper position and insert new node
	insert_helper(name, ID, head);
}


AVL_Tree::Node* AVL_Tree::insert_helper(const std::string& name, const std::string& ID, Node* root) {
	/*
	*	Determines the correct position of the new data and inserts into tree
	*	Once inserted, tree is rebalanced as required
	*	Prints results (successful or unseccussful) to console on completion of insertion
	*/

	// Base case for reaching end of tree
	if (root == nullptr) {
		#if (DEBUG != 1)
		cout << "successful" << endl;
		#endif
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
		#if (DEBUG != 1)
		cout << "unsuccessful" << endl;
		#endif
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


unsigned int AVL_Tree::get_height(Node* root) const {
	/*
	*	Helper function to account for leaf nodes in determing height of branches/Nodes
	*/

	if (root == nullptr) {return 0;}
	return root->height;
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Rebalance methods~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
AVL_Tree::Node* AVL_Tree::rebalance(Node* root) {
	/*
	*	Gets the balance factor of the node in question
	*	Performs appropriate rotations to balance the tree
	*/

	short balance_factor = get_balance_factor(root);

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


short AVL_Tree::get_balance_factor(Node* root) const {
	/*
	*	Determines the balance factor of a specific node
	*/

	if (root == nullptr) {return 0;}
	return get_height(root->l_child) - get_height(root->r_child);
}


AVL_Tree::Node* AVL_Tree::left_rotation(Node* old_root) {
	/*
	*	Performs a left rotation on the specified node
	*/

	Node* new_root = old_root->r_child;
	old_root->r_child = new_root->l_child;

	// Assign parent to left child of new root
	if (new_root->l_child) {
		new_root->l_child->parent = old_root;
	}

	// Adjust parent/child relationship with new root and it's new parent
	if (old_root->parent) {
		// Check which child the old root was
		if (old_root->is_left_child()) {
			old_root->parent->l_child = new_root;
		} else {
			old_root->parent->r_child = new_root;
		}
	} 
	
	// If old root was the head
	else {
		head = new_root;
	}

	// Clean up final connections and update heights
	new_root->l_child = old_root;
	new_root->parent = old_root->parent;
	old_root->parent = new_root;
	old_root->height = 1 + max(get_height(old_root->l_child), get_height(old_root->r_child));
	new_root->height = 1 + max(get_height(new_root->l_child), get_height(new_root->r_child));

	return new_root;
}


AVL_Tree::Node* AVL_Tree::right_rotation(Node* old_root) {
	/*
	*	Performs a right rotation on the specified node
	*/

	Node* new_root = old_root->l_child;
	old_root->l_child = new_root->r_child;

	// Assign parent to left child of new root
	if (new_root->r_child) {
		new_root->r_child->parent = old_root;
	}

	// Adjust parent/child relationship with new root and it's new parent
	if (old_root->parent) {
		// Check which child the old root was
		if (old_root->is_left_child()) {
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


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Removal methods~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
void AVL_Tree::remove(const string& remove_ID) {
	/*
	*	Removes the node with the specified ID from the tree
	*	Rebalances as required
	*/

	Node* node_to_remove = search_ID(remove_ID);
	if (node_to_remove) {
		remove_node(node_to_remove);
		#if (DEBUG != 1)
		cout << "successful" << endl;
		#endif
	} else {
		#if (DEBUG != 1)
		cout << "unsuccessful" << endl;
		#endif
	}
}


void AVL_Tree::remove_node(Node* deleting_node) {
	/*
	*	Removes specified node and rebalances as required
	*/

	// Case no-children
	if (!deleting_node->l_child && !deleting_node->r_child) {
		delete_node_no_children(deleting_node);
	}
	// Case two-children
	else if (deleting_node->l_child && deleting_node->r_child) {
		delete_node_two_children(deleting_node);
	}
	// Case one-child
	else {
		delete_node_one_child(deleting_node);
	}
}


void AVL_Tree::delete_node_no_children(Node* deleting_node) {
	/*
	*	Delete node that has no children
	*/

	Node* parent = deleting_node->parent;
	transplant(deleting_node);
	if (parent) {
		delete_fixup(parent);
	}
}


void AVL_Tree::delete_node_one_child(Node* deleting_node) {
	/*
	*	Delete node that has one child
	*/

	Node* parent = deleting_node->parent;
	Node* replacing_node = (deleting_node->r_child) ? deleting_node->r_child : deleting_node->l_child;
	transplant(deleting_node, replacing_node);
	if (parent) {
		delete_fixup(parent);
	}
}


void AVL_Tree::delete_node_two_children(Node* deleting_node) {
	/*
	*	Delete node that has two children
	*/
	Node* replacing_node = get_successor(deleting_node);
	*deleting_node = *replacing_node;
	if (replacing_node->r_child) {
		delete_node_one_child(replacing_node);
	} else {
		delete_node_no_children(replacing_node);
	}
	//remove_node(replacing_node);
}


void AVL_Tree::transplant(Node* deleting_node, Node* replacing_node) {
	/*
	*	Swaps two nodes during deletion and actually deletes
	*/

	// If deleting root node
	if (!deleting_node->parent) {
		head = replacing_node;
	} else if (deleting_node->is_left_child()) {
		deleting_node->parent->l_child = replacing_node;
	} else {
		deleting_node->parent->r_child = replacing_node;
	}

	if (replacing_node) {
		replacing_node->parent = deleting_node->parent;
	}
	delete deleting_node;
}


void AVL_Tree::delete_fixup(Node* fixing_node) {
	/*
	*	Rebalance tree after deletion
	*/

	while (fixing_node) {
		fixing_node->height = 1 + max(get_height(fixing_node->l_child), get_height(fixing_node->r_child));

		if (get_balance_factor(fixing_node) > 1) {
			// Case left-left
			if (get_balance_factor(fixing_node->l_child) >= 0) {
				right_rotation(fixing_node);
			}
			// Case left-right
			else {
				left_rotation(fixing_node->l_child);
				right_rotation(fixing_node);
			}
		}

		else if (get_balance_factor(fixing_node) < -1) {
			// Case right-right
			if (get_balance_factor(fixing_node->r_child) <= 0) {
				left_rotation(fixing_node);
			}
			//C Case right-left
			else {
				right_rotation(fixing_node->r_child);
				left_rotation(fixing_node);
			}
		}

		fixing_node = fixing_node->parent;
	}
}


AVL_Tree::Node* AVL_Tree::get_successor(Node* const root) const {
	/*
	*	Finds the immediate successor of specified node
	*/

	Node* curr = nullptr;
	// If successor is in root's tree
	if (root->r_child) {
		curr = root->r_child;
		while(curr->l_child) {
			curr = curr->l_child;
		}
		return curr;
	}

	// If successor is ancestor
	Node* parent_node = root->parent;
	curr = root;
	while (parent_node) {
		if (curr->ID == parent_node->l_child->ID) {
			break;
		}
		curr = parent_node;
		parent_node = parent_node->parent;
	}
	return parent_node;
}


void AVL_Tree::removeInOrder(const unsigned int N) {
	/*
	*	Removes the Nth in order node from the tree
	*	Rebalances as required
	*/

	// Get in order vector of nodes
	vector<Node*> nodes;
	in_order_helper(head, nodes);

	// Check if Nth node exists
	if (N >= nodes.size()) {
		#if (DEBUG != 1)
		cout << "unsuccessful" << endl;
		#endif
	}

	// Find Nth node and remove
	remove_node(nodes[N]);
	#if (DEBUG != 1)
    cout << "successful" << endl;
	#endif
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Search methods~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
void AVL_Tree::search(const string& search_item) const {
	/*
	*	User facing search method
	*	Determines what is being searched for and calls appropriate helper function
	*/

	if (is_number(search_item)) {
		Node* node = search_ID(search_item);
		if (node) {cout << node->name << endl;}
	} else {
		search_name(search_item);
	}
}


bool AVL_Tree::is_number(const string& item) const {
	/*
	*	Determines if given string is a number (ID) or not
	*/

	return !item.empty() && all_of(item.begin(), item.end(), ::isdigit);
}


AVL_Tree::Node* AVL_Tree::search_ID(const string& search_ID) const {
	/*
	*	Helper function to search for a node with a specified ID
	*	If ID is found, returns node with that ID
	*	If ID is not found, prints unsuccessful and return nullptr
	*/

	Node* curr = head;
	while (curr) {
		if (curr->ID == search_ID) {
			return curr;
		}
		curr = (search_ID < curr->ID) ? curr->l_child : curr->r_child;
	}
	cout << "unsuccessful" << endl;	
	return nullptr;
}


void AVL_Tree::search_name(const string& search_name) const {
	/*
	*	Helper function to search for all nodes with a specified name
	*	Prints the all ID's associated with the name, or unsuccessful if name was not found;
	*/

	vector<Node*> nodes;
	bool found{false};
	pre_order_helper(head, nodes);
	for (Node* node: nodes) {
		if (node->name == search_name) {
			found = true;
			cout << node->ID << endl;
		}
	}

	if (!found) {
		#if (DEBUG != 1)
		cout << "unsuccessful" << endl;
		#endif
	}
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Traversal methods~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
vector<string> AVL_Tree::InOrderTraversal(const string& val) const {
	/*
	*	Return in order vector of nodes
	*	Designed for testing
	*/

	vector<Node*> nodes;
	in_order_helper(head, nodes);
	
	vector<string> result;
	if (val == "name") {
		for (const Node* node: nodes) {
			result.push_back(node->name);
		}
	} else if (val == "ID") {
		for (const Node* node: nodes) {
			result.push_back(node->ID);
		}
	}
	return result;
}


void AVL_Tree::in_order_helper(Node* const root, vector<Node*>& nodes) const {
	/*
	*	Recursive helper function for in order print
	*/

	if (root) {
		in_order_helper(root->l_child, nodes);
		nodes.push_back(root);
		in_order_helper(root->r_child, nodes);
	}
}


vector<string> AVL_Tree::PreOrderTraversal(const string& val) const {
	/*
	*	Return in order vector of nodes
	*	Designed for testing
	*/

	vector<Node*> nodes;
	pre_order_helper(head, nodes);
	
	vector<string> result;
	if (val == "name") {
		for (const Node* node: nodes) {
			result.push_back(node->name);
		}
	} else if (val == "ID") {
		for (const Node* node: nodes) {
			result.push_back(node->ID);
		}
	}
	return result;
}


void AVL_Tree::pre_order_helper(Node* const root, vector<Node*>& nodes) const {
	/*
	*	Recursive helper function for pre order print
	*/

	if (root) {
		nodes.push_back(root);
		pre_order_helper(root->l_child, nodes);
		pre_order_helper(root->r_child, nodes);
	}
}


vector<string> AVL_Tree::PostOrderTraversal(const string& val) const {
	/*
	*	Return in order vector of nodes
	*	Designed for testing
	*/

	vector<Node*> nodes;
	post_order_helper(head, nodes);
	
	vector<string> result;
	if (val == "name") {
		for (const Node* node: nodes) {
			result.push_back(node->name);
		}
	} else if (val == "ID") {
		for (const Node* node: nodes) {
			result.push_back(node->ID);
		}
	}
	return result;
}


void AVL_Tree::post_order_helper(Node* const root, vector<Node*>& nodes) const {
	/*
	*	Recursive helper function for post order print
	*/

	if (root) {
		post_order_helper(root->l_child, nodes);
		post_order_helper(root->r_child, nodes);
		nodes.push_back(root);
	}
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Other methods~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
void AVL_Tree::printLevelCount() const {
	/*
	*	Prints the current height of the tree
	*/

	if (!head) {
		cout << 0 << endl;
	}
	cout << head->height << endl;
}


bool AVL_Tree::empty() const {
	return head == nullptr;
}