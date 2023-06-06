#include "AVL_Interface.h"
using namespace std;

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
		commandSwitchboard(commands);
	}
}


bool AVL_Interface::isValidName(const string& str) const {
	/*
	*	Checks for valid name, allows only letters and spaces
	*	Must be enclosed in double quotes
	*/

	return str.length() > 0 && regex_match(str, regex("^\"[A-Z a-z]+\"$"));
}


bool AVL_Interface::isValidID(const string& str) const {
	/*
	*	Checks for valid ID, allows only numbers and must be 8 digits long
	*/

	return regex_match(str, regex("^[0-9]{8}$"));
}


void AVL_Interface::commandSwitchboard(const vector<string>& commands) {
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
		printInOrder();
	} else if (commands[0] == "printPreorder") {
		printPreOrder();
	} else if (commands[0] == "printPostorder") {
		printPostOrder();
	} else if (commands[0] == "printLevelCount") {
		tree.printLevelCount();
	} else if (isValidRemoveNth(commands)) {
		tree.removeInOrder(stoi(commands[1]));
	} else {
		// If invalid
		#if (DEBUG != 1)
		cout << "unsuccessful" << endl;
		#endif
	}
}


bool AVL_Interface::isValidInsert(const vector<string>& commands) const {
	/*
	*	Helper function to abstract away checking for valid insert command
	*/

	return commands.size() == 3 && commands[0] == "insert" && isValidName(commands[1]) && isValidID(commands[2]);
}


bool AVL_Interface::isValidRemove(const vector<string>& commands) const {
	/*
	*	Helper function to abstract away checking for valid remove command
	*/

	return commands.size() == 2 && commands[0] == "remove" && isValidID(commands[1]);
}


bool AVL_Interface::isValidSearch(const vector<string>& commands) const {
	/*
	*	Helper function to abstract away checking for valid search command
	*/

	return commands.size() == 2 && commands[0] == "search" && (isValidName(commands[1]) || isValidID(commands[1]));
}


bool AVL_Interface::isValidRemoveNth(const vector<string>& commands) const {
	/*
	*	Helper function to abstract away checking for valid insert command
	*/

	return commands.size() == 2 && commands[0] == "removeInorder" && commands[1] != "" && all_of(commands[1].begin(), commands[1].end(), ::isdigit);
}


string AVL_Interface::stripQuotes(const string& name) const {
	/*
	*	Strips double quotes from name inputs
	*/	
	string newName = name;
	newName.erase(remove(newName.begin(), newName.end(), '"'), newName.end());
	return newName;
}


void AVL_Interface::printInOrder() const {
	/*
	*	In order print of all names in the tree, based on ID
	*/

    if (!tree.empty()) {
        cout << endl;;
        return;
    }
	vector<string> names = tree.InOrderTraversal();
	print_helper(names);
}


void AVL_Interface::printPreOrder() const {
	/*
	*	Pre order print of all names in the tree, based on ID
	*/

    if (!tree.empty()) {
        cout << endl;;
        return;
    }
	vector<string> names = tree.PreOrderTraversal();
	print_helper(names);
}


void AVL_Interface::printPostOrder() const {
	/*
	*	Pre order print of all names in the tree, based on ID
	*/

    if (!tree.empty()) {
        cout << endl;;
        return;
    }
	vector<string> names = tree.PostOrderTraversal();
	print_helper(names);
}


void AVL_Interface::print_helper(const vector<string>& items) const {
	/*
	*	Helper function to print results of various traversals
	*/

	for (int i{}; i < items.size(); ++i) {
		if (i != 0) {
			cout << ", ";
		}
		cout << items[i];
	}

	cout << endl;
}