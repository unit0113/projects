#pragma once
#include <regex>
#include <sstream>
#include "AVL_Tree.h"

class AVL_Interface {
	public:
		AVL_Tree tree;

		// Helper Functions
		bool isValidName(const std::string& str);
		bool isValidID(const std::string& str);
		void commandSwitchboard(AVL_Tree& tree, const std::vector<std::string>& commands);
		bool isValidInsert(const std::vector<std::string>& commands);
		bool isValidRemove(const std::vector<std::string>& commands);
		bool isValidSearch(const std::vector<std::string>& commands);
		bool isValidRemoveNth(const std::vector<std::string>& commands);
		std::string stripQuotes(std::string name);

	public:
		void run();

};