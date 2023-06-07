#pragma once
#include <regex>
#include <sstream>
#include "AVL_Tree.h"

class AVL_Interface {
	private:
		AVL_Tree tree;

		// Helper Functions
		bool isValidName(const std::string& str) const;
		bool isValidID(const std::string& str) const;
		void commandSwitchboard(const std::vector<std::string>& commands);
		bool isValidInsert(const std::vector<std::string>& commands) const;
		bool isValidRemove(const std::vector<std::string>& commands) const;
		bool isValidSearch(const std::vector<std::string>& commands) const;
		bool isValidRemoveNth(const std::vector<std::string>& commands) const;
		std::string stripQuotes(const std::string& name) const;
		void printInOrder() const;
		void printPreOrder() const;
		void printPostOrder() const;
		void print_helper(const std::vector<std::string>& nodes) const;

	public:
		void run();

};