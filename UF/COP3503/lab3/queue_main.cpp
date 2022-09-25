#include <iostream>
#include "ABQ.h"
using namespace std;

int main()
{
cout << "Making integer ABQ...\n";
	ABQ<int> intABQ;
	cout << "Size: " << intABQ.getSize() << endl;
	cout << "Max Capacity: " << intABQ.getMaxCapacity() << endl;
	cout << "\nEnqueueing items...\n";
	for (int i = 1; i < 10; i++)
	{
		intABQ.enqueue(i);
		cout << "\nEnqueued " << i << endl;
		cout << "New Size: " << intABQ.getSize() << endl;
		cout << "New Max Capacity: " << intABQ.getMaxCapacity() << endl;
	}

	cout << "\nDequeueing items...\n";
	for (int i = 1; i < 10; i++)
	{
		cout << "\nDequeued " << intABQ.dequeue() << endl;
		cout << "New Size: " << intABQ.getSize() << endl;
		cout << "New Max Capacity: " << intABQ.getMaxCapacity() << endl;
	}


	cout << "\nMaking float ABQ...\n";
	ABQ<float> floatABQ(10);
	cout << "Size: " << floatABQ.getSize() << endl;
	cout << "Max Capacity: " << floatABQ.getMaxCapacity() << endl;
	cout << "\nEnqueueing items...\n";
	for (float i = 1; i < 5; i += 0.5f)
	{
		floatABQ.enqueue(i);
		cout << "\nEnqueued " << i << endl;
		cout << "New Size: " << floatABQ.getSize() << endl;
		cout << "New Max Capacity: " << floatABQ.getMaxCapacity() << endl;
	}

	cout << "\nDequeueing items...\n";
	for (float i = 1; i < 5; i += 0.5f)
	{
		cout << "\nDequeued " << floatABQ.dequeue() << endl;
		cout << "New Size: " << floatABQ.getSize() << endl;
		cout << "New Max Capacity: " << floatABQ.getMaxCapacity() << endl;
	}
	
	return 0;
}