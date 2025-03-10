#include "../src/sum_of_level.h"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

/*
	To check output (At the sum_of_level directory):
		g++ -std=c++14 -Werror -Wuninitialized -o test test-unit/test.cpp && ./test
*/

TEST_CASE("Function: sum of level 1", "[given]") {
	TreeNode* root = nullptr;
	root = insert(root, 2);
	root = insert(root, 1);
	root = insert(root, 3);
	root = insert(root, 4);
	REQUIRE(2 == sumOfLevel(root, 0));
}


TEST_CASE("Function: sum of level 2", "[given]") {
	TreeNode* root = nullptr;
	root = insert(root, 2);
	root = insert(root, 1);
	root = insert(root, 3);
	root = insert(root, 4);
	REQUIRE(4 == sumOfLevel(root, 1));
}

TEST_CASE("Function: sum of level 3", "[given]") {
	TreeNode* root = nullptr;
	root = insert(root, 2);
	root = insert(root, 1);
	root = insert(root, 3);
	root = insert(root, 4);
	REQUIRE(4 == sumOfLevel(root, 2));
}

TEST_CASE("Function: sum of level 4", "[given]") {
	TreeNode* root = nullptr;
	root = insert(root, 2);
	root = insert(root, 1);
	root = insert(root, 3);
	root = insert(root, 4);
	REQUIRE(0 == sumOfLevel(root, 3));
}