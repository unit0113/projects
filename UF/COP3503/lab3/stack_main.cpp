#include <iostream>
#include "ABS.h"
using namespace std;

int main()
{
	cout << "Making integer ABS...\n";
	ABS<int> intABS;
	cout << "Size: " << intABS.getSize() << endl;
	cout << "Max Capacity: " << intABS.getMaxCapacity() << endl;
	cout << "\nPushing items...\n";
	for (int i = 1; i < 10; i++)
	{
		intABS.push(i);
		cout << "\nPushed " << intABS.peek() << endl;
		cout << "New Size: " << intABS.getSize() << endl;
		cout << "New Max Capacity: " << intABS.getMaxCapacity() << endl;
	}

	cout << "\nPopping items...\n";
	for (int i = 1; i < 10; i++)
	{
		cout << "\nPopped " << intABS.pop() << endl;
		cout << "New Size: " << intABS.getSize() << endl;
		cout << "New Max Capacity: " << intABS.getMaxCapacity() << endl;
	}


	cout << "\nMaking float ABS...\n";
	ABS<float> floatABS(10);
	cout << "Size: " << floatABS.getSize() << endl;
	cout << "Max Capacity: " << floatABS.getMaxCapacity() << endl;
	cout << "\nPushing items...\n";
	for (float i = 1; i < 5; i += 0.5f)
	{
		floatABS.push(i);
		cout << "\nPushed " << floatABS.peek() << endl;
		cout << "New Size: " << floatABS.getSize() << endl;
		cout << "New Max Capacity: " << floatABS.getMaxCapacity() << endl;
	}

	cout << "\nPopping items...\n";
	for (float i = 1; i < 5; i += 0.5f)
	{
		cout << "\nPopped " << floatABS.pop() << endl;
		cout << "New Size: " << floatABS.getSize() << endl;
		cout << "New Max Capacity: " << floatABS.getMaxCapacity() << endl;
	}
	
	return 0;
}