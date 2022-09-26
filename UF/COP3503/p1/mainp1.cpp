#include <iostream>
#include <string>
#include <sstream>
#include "LinkedList.h"
using namespace std;

void TestAddHead();
void TestAddTail();
void TestAddingArrays();

int main()
{
	int testNum;
	cin >> testNum;
	if (testNum == 1)
		TestAddHead();
	else if (testNum == 2)
		TestAddTail();
	else if (testNum == 3)
		TestAddingArrays();

	return 0;
}

void TestAddHead()
{
	cout << "=====Testing AddHead() functionality====" << endl;
	LinkedList<int> data;
	for (int i = 0; i < 12; i += 2)
		data.AddHead(i);
	cout << "Node count: " << data.NodeCount() << endl;
	cout << "Print list forward:" << endl;
	data.PrintForward();
	cout << "Print list in reverse:" << endl;
	data.PrintReverse();
}

void TestAddTail()
{
	cout << "=====Testing AddTail() functionality====" << endl;
	LinkedList<int> data;
	for (int i = 0; i <= 21; i += 3)
		data.AddTail(i);
	cout << "Node count: " << data.NodeCount() << endl;
	cout << "Print list forward:" << endl;
	data.PrintForward();
	cout << "Print list in reverse:" << endl;
	data.PrintReverse();
}

void TestAddingArrays()
{
	cout << "=====Testing AddNodesHead() and AddNodesTail() =====" << endl;

	string values[5];
	values[0] = "*";
	values[1] = "**";
	values[2] = "***";
	values[3] = "****";
	values[4] = "*****";

	LinkedList<string> list;
	list.AddHead("**");
	list.AddHead("***");
	list.AddHead("****");
	list.AddHead("*****");
	list.AddHead("******");
	list.AddHead("*******");
	list.AddHead("********");
	list.AddHead("*********");
	list.AddHead("********");
	list.AddHead("*******");
	list.AddHead("******");

	list.AddNodesHead(values, 5);
	list.AddNodesTail(values, 5);
	list.PrintForward();
}