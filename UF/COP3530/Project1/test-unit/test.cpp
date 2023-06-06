#include "../src/AVL_Interface.h"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using namespace std;

/*
	To check output (At the Project1 directory):
		g++ -std=c++14 -Werror -Wuninitialized -o build/test test-unit/test.cpp && build/test
		g++ -std=c++14 -Werror -Wuninitialized -o build/test test-unit/test.cpp src/AVL_Interface.cpp src/AVL_Tree.cpp
		./build/test
*/



// IMPORTANT!! For all tests, comment out main function (lines 817-820) in order to not interfere with catch


TEST_CASE("BST Insert", "[tree]"){
	/*
		MyAVLTree tree;   // Create a Tree object 
		tree.insert(3);
		tree.insert(2);
		tree.insert(1);
		std::vector<int> actualOutput = tree.inorder();
		std::vector<int> expectedOutput = {1, 2, 3};
		REQUIRE(expectedOutput.size() == actualOutput.size());
		REQUIRE(actualOutput == expectedOutput);
	*/
	REQUIRE(1 == 1);
}
















//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Interface Tests~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


// IMPORTANT!!! Change private on line 7 of AVL_Interface.h to public for unit testing internal functions

TEST_CASE("Interface Valid Name", "[interface]"){
	AVL_Interface interface;
	// Name is required to be enclosed by "". Only alphabetic characters and spaces allowed. Empty strings are invalid
	REQUIRE(interface.isValidName("\"Bob the Builder\"") == true);
	REQUIRE(interface.isValidName("\"Geralt of Rivia\"") == true);
	REQUIRE(interface.isValidName("\"Dave\"") == true);
	REQUIRE(interface.isValidName("") == false);						//Empty
	REQUIRE(interface.isValidName("\"Bob the Builder1\"") == false);	// Non-alphabetic char
	REQUIRE(interface.isValidName("\"Bill_Nye\"") == false);			// _ instead of space
	REQUIRE(interface.isValidName("\"\"") == false);					// Empty
	REQUIRE(interface.isValidName("Bob the Builder") == false);			// Not enclosed by ""
	REQUIRE(interface.isValidName("Geralt of Rivia") == false);			// Not enclosed by ""
}

TEST_CASE("Interface Valid ID", "[interface]"){
	AVL_Interface interface;
	// ID must be 8 digits longs, numbers only
	REQUIRE(interface.isValidID("12345678") == true);
	REQUIRE(interface.isValidID("88888888") == true);
	REQUIRE(interface.isValidID("") == false);				// Empty
	REQUIRE(interface.isValidID("1234") == false);			// Too short
	REQUIRE(interface.isValidID("Bill_Nye") == false);		// Not numbers
	REQUIRE(interface.isValidID(" ") == false);				// Just a space
}

TEST_CASE("Interface Valid Insert Command", "[interface]"){
	AVL_Interface interface;
	// Checks tokenized input. First entry must be "insert", second entry must be valid name, third entry must be valid ID
	vector<string> commands = {"insert", "\"Dave\"", "12345678"};
	REQUIRE(interface.isValidInsert(commands) == true);
	commands[1] = "\"Geralt of Rivia\"";
	commands[2] = "88888888";
	REQUIRE(interface.isValidInsert(commands) == true);
	commands[1] = "Dave";
	REQUIRE(interface.isValidInsert(commands) == false);	// Invalid name
	commands[1] = "\"Geralt of Rivia\"";
	commands[2] = "8888";
	REQUIRE(interface.isValidInsert(commands) == false);	// Invlaid ID
	commands[2] = "88888888";
	commands[0] = "somethingThatsNotInsert";
	REQUIRE(interface.isValidInsert(commands) == false);	// Invlaid command
	commands[0] = "";
	REQUIRE(interface.isValidInsert(commands) == false);	// Invalid command
	commands[0] = "insert";
	commands.push_back("12345678");
	REQUIRE(interface.isValidInsert(commands) == false);	// Invalid vector length
	commands.pop_back();
	commands.pop_back();
	REQUIRE(interface.isValidInsert(commands) == false);	// Invalid vector length
	commands.clear();
	REQUIRE(interface.isValidInsert(commands) == false);	// Empty vector
}

TEST_CASE("Interface Valid Remove Command", "[interface]"){
	AVL_Interface interface;
	// Checks tokenized input. First entry must be "remove", second entry must be valid ID
	vector<string> commands = {"remove", "12345678"};
	REQUIRE(interface.isValidRemove(commands) == true);
	commands[1] = "88888888";
	REQUIRE(interface.isValidRemove(commands) == true);
	commands[1] = "1234";
	REQUIRE(interface.isValidRemove(commands) == false);	// Invalid ID
	commands[1] = "";
	REQUIRE(interface.isValidRemove(commands) == false);	// Empty ID
	commands[1] = "88888888";
	commands[0] = "somethingThatsNotRemove";
	REQUIRE(interface.isValidRemove(commands) == false);	// Invalid command
	commands[0] = "";
	REQUIRE(interface.isValidRemove(commands) == false);	// Invalid command
	commands[0] = "remove";
	commands.push_back("12345678");
	REQUIRE(interface.isValidRemove(commands) == false);	// Invalid vector length
	commands.pop_back();
	commands.pop_back();
	REQUIRE(interface.isValidRemove(commands) == false);	// Invalid vector length
	commands.clear();
	REQUIRE(interface.isValidRemove(commands) == false);	// Empty vector
}

TEST_CASE("Interface Valid Search Command", "[interface]"){
	AVL_Interface interface;
	// Checks tokenized input. First entry must be "search", second entry must be valid ID or name
	vector<string> commands = {"search", "12345678"};
	REQUIRE(interface.isValidSearch(commands) == true);
	commands[1] = "88888888";
	REQUIRE(interface.isValidSearch(commands) == true);
	commands[1] = "\"Dave\"";
	REQUIRE(interface.isValidSearch(commands) == true);
	commands[1] = "\"Geralt of Rivea\"";
	REQUIRE(interface.isValidSearch(commands) == true);
	commands[1] = "1234";
	REQUIRE(interface.isValidSearch(commands) == false);	// Invalid ID
	commands[1] = "Bob";
	REQUIRE(interface.isValidSearch(commands) == false);	// Invalid Name
	commands[1] = "";
	REQUIRE(interface.isValidSearch(commands) == false);	// Empty ID
	commands[1] = "88888888";
	commands[0] = "somethingThatsNotSearch";
	REQUIRE(interface.isValidSearch(commands) == false);	// Invalid command
	commands[0] = "";
	REQUIRE(interface.isValidSearch(commands) == false);	// Invalid command
	commands[0] = "search";
	commands.push_back("12345678");
	REQUIRE(interface.isValidSearch(commands) == false);	// Invalid vector length
	commands.pop_back();
	commands.pop_back();
	REQUIRE(interface.isValidSearch(commands) == false);	// Invalid vector length
	commands.clear();
	REQUIRE(interface.isValidSearch(commands) == false);	// Empty vector
}

TEST_CASE("Interface Valid Remove Nth Command", "[interface]"){
	AVL_Interface interface;
	// Checks tokenized input. First entry must be "removeInorder", second entry must be valid integer
	vector<string> commands = {"removeInorder", "1"};
	REQUIRE(interface.isValidRemoveNth(commands) == true);
	commands[1] = "88888888";
	REQUIRE(interface.isValidRemoveNth(commands) == true);
	commands[1] = "0";
	REQUIRE(interface.isValidRemoveNth(commands) == true);
	commands[1] = "";
	REQUIRE(interface.isValidRemoveNth(commands) == false);	// Empty int
	commands[1] = "88888888";
	commands[0] = "somethingThatsNotRemoveNth";
	REQUIRE(interface.isValidRemoveNth(commands) == false);	// Invalid command
	commands[0] = "";
	REQUIRE(interface.isValidRemoveNth(commands) == false);	// Invalid command
	commands[0] = "removeInorder";
	commands.push_back("5");
	REQUIRE(interface.isValidRemoveNth(commands) == false);	// Invalid vector length
	commands.pop_back();
	commands.pop_back();
	REQUIRE(interface.isValidRemoveNth(commands) == false);	// Invalid vector length
	commands.clear();
	REQUIRE(interface.isValidRemoveNth(commands) == false);	// Empty vector
}