#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <regex>
#include <sstream>

using namespace std;

/* Note: 
	1. You will have to comment main() when unit testing your code because catch uses its own main().
	2. You will submit this main.cpp file and any header files you have on Gradescope. 
*/

class AVL_Tree {
	/*
	*	Balanced AVL binary search tree
	*	Support insertion, search, deletion, delete Nth, and traversals
	*/

	private:
	
		struct Node {
			/*
			*	Private node struct for use in tree
			*/
			string name;
			string ID;
			unsigned int height = 1;
			Node* parent = nullptr;
			Node* l_child = nullptr;
			Node* r_child = nullptr;

			Node(string new_name, string new_ID, Node* new_parent=nullptr) :
				name(new_name),
				ID(new_ID),
				parent(new_parent) {}
			Node& operator=(const Node& other);
			bool is_left_child() const;
		};

		Node* head = nullptr;

		// Helper methods
		Node* insert_helper(string name, string ID, Node* root);
		unsigned int get_height(Node* root) const;
		Node* rebalance(Node* root);
		short get_balance_factor(Node* root) const;
		Node* left_rotation(Node* old_root);
		Node* right_rotation(Node* old_root);
		// Seach helpers
		bool is_number(const string& item) const;
		Node* search_ID(const string& search_ID) const;
		void search_name(const string& search_name) const;
		// Traversal helpers
		void in_order_helper(Node* root, vector<Node*>& nodes) const;
		void print_helper(vector<Node*>& nodes) const;
		void pre_order_helper(Node* root, vector<Node*>& nodes) const;
		void post_order_helper(Node* root, vector<Node*>& nodes) const;
		// Removal helpers
		void remove_node(Node* deleting_node);
		void delete_node_no_children(Node* deleting_node);
		void delete_node_one_child(Node* deleting_node);
		void delete_node_two_children(Node* deleting_node);
		void transplant(Node* deleting_node, Node* replacing_node=nullptr);
		void delete_fixup(Node* fixing_node);
		Node* get_successor(Node* root);

	public:
		~AVL_Tree();
		void insert(string name, string ID);
		void remove(const string& remove_ID);
		void search(const string& search_item) const;
		void printInOrder() const;
		void printPreOrder() const;
		void printPostOrder() const;
		void printLevelCount() const;
		void removeInOrder(unsigned int N);
		vector<string> InOrderTraversal(string val) const;
};


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
void AVL_Tree::insert(string name, string ID) {
	/*
	*	Inserts new item into tree
	*/

	// For empty tree
	if (head == nullptr) {
		head = new Node(name, ID);
		cout << "successful" << endl;
		return;
	}

	// Else find proper position and insert new node
	insert_helper(name, ID, head);
}


AVL_Tree::Node* AVL_Tree::insert_helper(string name, string ID, Node* root) {
	/*
	*	Determines the correct position of the new data and inserts into tree
	*	Once inserted, tree is rebalanced as required
	*	Prints results (successful or unseccussful) to console on completion of insertion
	*/

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
		cout << "successful" << endl;
	} else {
		cout << "unsuccessful" << endl;
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
	if (parent) {//move to end of delete? or start of delete fixup?
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
	remove_node(replacing_node);
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


AVL_Tree::Node* AVL_Tree::get_successor(Node* root) {
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


void AVL_Tree::removeInOrder(unsigned int N) {
	/*
	*	Removes the Nth in order node from the tree
	*	Rebalances as required
	*/

	// Get in order vector of nodes
	vector<Node*> nodes;
	in_order_helper(head, nodes);

	// Check if Nth node exists
	if (N >= nodes.size()) {
		cout << "unsuccessful" << endl;
	}

	// Find Nth node and remove
	remove_node(nodes[N]);
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
		cout << "unsuccessful" << endl;
	}
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Traversal methods~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
void AVL_Tree::printInOrder() const {
	/*
	*	In order print of all names in the tree, based on ID
	*/

	vector<Node*> nodes;
	in_order_helper(head, nodes);
	print_helper(nodes);
}


void AVL_Tree::in_order_helper(Node* root, vector<Node*>& nodes) const {
	/*
	*	Recursive helper function for in order print
	*/

	if (root) {
		in_order_helper(root->l_child, nodes);
		nodes.push_back(root);
		in_order_helper(root->r_child, nodes);
	}
}


vector<string> AVL_Tree::InOrderTraversal(string val) const {
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


void AVL_Tree::printPreOrder() const {
	/*
	*	Pre order print of all names in the tree, based on ID
	*/

	vector<Node*> nodes;
	pre_order_helper(head, nodes);
	print_helper(nodes);
}


void AVL_Tree::pre_order_helper(Node* root, vector<Node*>& nodes) const {
	/*
	*	Recursive helper function for pre order print
	*/

	if (root) {
		nodes.push_back(root);
		pre_order_helper(root->l_child, nodes);
		pre_order_helper(root->r_child, nodes);
	}
}

		
void AVL_Tree::printPostOrder() const {
	/*
	*	Post order print of all names in the tree, based on ID
	*/

	vector<Node*> nodes;
	post_order_helper(head, nodes);
	print_helper(nodes);
}


void AVL_Tree::post_order_helper(Node* root, vector<Node*>& nodes) const {
	/*
	*	Recursive helper function for post order print
	*/

	if (root) {
		post_order_helper(root->l_child, nodes);
		post_order_helper(root->r_child, nodes);
		nodes.push_back(root);
	}
}


void AVL_Tree::print_helper(vector<Node*>& nodes) const {
	/*
	*	Helper function to print results of various traversals
	*/

	for (int i{}; i < nodes.size(); ++i) {
		if (i != 0) {
			cout << ", ";
		}
		cout << nodes[i]->name;
	}

	cout << endl;
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



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Interface~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
class AVL_Interface {
	public:
		AVL_Tree tree;

		// Helper Functions
		bool isValidName(const string& str);
		bool isValidID(const string& str);
		void commandSwitchboard(AVL_Tree& tree, const vector<string>& commands);
		bool isValidInsert(const vector<string>& commands);
		bool isValidRemove(const vector<string>& commands);
		bool isValidSearch(const vector<string>& commands);
		bool isValidRemoveNth(const vector<string>& commands);
		string stripQuotes(string name);

	public:
		void run();

};


void AVL_Interface::run() {
	/*
	*	Main interface between AVL tree and user
	*	Gets input from user and calls appropriate AVL tree functions
	*/

	// Get number of commands
	int num_commands{};
	cin >> num_commands;
	
	// Clear new line character
	cin.ignore();

	// Read and store commands
	vector<string> commands;
	string line, word;
	for (int i{}; i < num_commands; ++i) {
		commands.clear();

		// Tokenize input
		getline(cin, line);
		istringstream ss(line);
		while (getline(ss, word, ' ')) {
			commands.push_back(word);
		}

		// Parse input and call appropriate command in tree
		commandSwitchboard(tree, commands);
	}
}


bool AVL_Interface::isValidName(const string& str) {
	/*
	*	Checks for valid name, allows only letters and spaces
	*	Must be enclosed in double quotes
	*/

	return str.length() > 0 && regex_match(str, regex("^\"[A-Z a-z]+\"$"));
}


bool AVL_Interface::isValidID(const string& str) {
	/*
	*	Checks for valid ID, allows only numbers and must be 8 digits long
	*/

	return regex_match(str, regex("^[0-9]{8}$"));
}


void AVL_Interface::commandSwitchboard(AVL_Tree& tree, const vector<string>& commands) {
	/*
	*	Checks for valid inputs and calls appropriate commands
	*	Prints new line if input is invalid
	*/
	if (isValidInsert(commands)) {
		tree.insert(stripQuotes(commands[1]), commands[2]);
	} else if (isValidRemove(commands)) {
		tree.remove(commands[1]);
	} else if (isValidSearch(commands)) {
		tree.search(stripQuotes(commands[1]));
	} else if (commands[0] == "printInorder") {
		tree.printInOrder();
	} else if (commands[0] == "printPreorder") {
		tree.printPreOrder();
	} else if (commands[0] == "printPostorder") {
		tree.printPostOrder();
	} else if (commands[0] == "printLevelCount") {
		tree.printLevelCount();
	} else if (isValidRemoveNth(commands)) {
		tree.removeInOrder(stoi(commands[1]));
	} else {
		// If invalid
		cout << "unsuccessful" << endl;
	}
}


bool AVL_Interface::isValidInsert(const vector<string>& commands) {
	/*
	*	Helper function to abstract away checking for valid insert command
	*/

	return commands.size() == 3 && commands[0] == "insert" && isValidName(commands[1]) && isValidID(commands[2]);
}


bool AVL_Interface::isValidRemove(const vector<string>& commands) {
	/*
	*	Helper function to abstract away checking for valid remove command
	*/

	return commands.size() == 2 && commands[0] == "remove" && isValidID(commands[1]);
}


bool AVL_Interface::isValidSearch(const vector<string>& commands) {
	/*
	*	Helper function to abstract away checking for valid search command
	*/

	return commands.size() == 2 && commands[0] == "search" && (isValidName(commands[1]) || isValidID(commands[1]));
}


bool AVL_Interface::isValidRemoveNth(const vector<string>& commands) {
	/*
	*	Helper function to abstract away checking for valid insert command
	*/

	return commands.size() == 2 && commands[0] == "removeInorder" && commands[1] != "" && all_of(commands[1].begin(), commands[1].end(), ::isdigit);
}


string AVL_Interface::stripQuotes(string name) {
	/*
	*	Strips double quotes from name inputs
	*/	
	string newName = name;
	newName.erase(remove(newName.begin(), newName.end(), '"'), newName.end());
	return newName;
}


int main(){
	AVL_Interface interface;
	interface.run();
}
