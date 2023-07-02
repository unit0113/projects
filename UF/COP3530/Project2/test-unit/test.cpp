#include "../src/pageRank.h"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include <iostream>
#include <string>

using namespace std;


/*
	To check output (At the Project1 directory):
		g++ -std=c++14 -Werror -Wuninitialized -o build/test test-unit/test.cpp src/pageRank.cpp src/adjacencyList.cpp
		./build/test
*/


TEST_CASE("BST Basic Insert", "[tree]"){
	REQUIRE("thing1" == "thing2");
}
