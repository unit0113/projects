#pragma once
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>


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
			std::string name;
			std::string ID;
			unsigned int height = 1;
			Node* parent = nullptr;
			Node* l_child = nullptr;
			Node* r_child = nullptr;

			Node(std::string new_name, std::string new_ID, Node* new_parent=nullptr) :
				name(new_name),
				ID(new_ID),
				parent(new_parent) {}
			Node& operator=(const Node& other);
			bool is_left_child() const;
		};

		Node* head = nullptr;

		// Helper methods
		Node* insert_helper(const std::string& name, const std::string& ID, Node* root);
		unsigned int get_height(Node* root) const;
		Node* rebalance(Node* root);
		short get_balance_factor(Node* root) const;
		Node* left_rotation(Node* old_root);
		Node* right_rotation(Node* old_root);
		// Seach helpers
		bool is_number(const std::string& item) const;
		Node* search_ID(const std::string& search_ID) const;
		void search_name(const std::string& search_name) const;
		// Traversal helpers
		void in_order_helper(Node* const root, std::vector<Node*>& nodes) const;
		void pre_order_helper(Node* const root, std::vector<Node*>& nodes) const;
		void post_order_helper(Node* const root, std::vector<Node*>& nodes) const;
		// Removal helpers
		void remove_node(Node* deleting_node);
		void delete_node_no_children(Node* deleting_node);
		void delete_node_one_child(Node* deleting_node);
		void delete_node_two_children(Node* deleting_node);
		void transplant(Node* deleting_node, Node* replacing_node=nullptr);
		void delete_fixup(Node* fixing_node);
		Node* get_successor(Node* const root) const;

	public:
		~AVL_Tree();
		void insert(const std::string& name, const std::string& ID);
		void remove(const std::string& remove_ID);
		void search(const std::string& search_item) const;
		void printLevelCount() const;
		void removeInOrder(const unsigned int N);
		bool empty() const;
		std::vector<std::string> InOrderTraversal(const std::string& val="name") const;
		std::vector<std::string> PreOrderTraversal(const std::string& val="name") const;
		std::vector<std::string> PostOrderTraversal(const std::string& val="name") const;
};