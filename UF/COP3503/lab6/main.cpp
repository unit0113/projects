// main.cpp : Defines the entry point for the console application.
//
#include <iostream>
#include <iomanip>
#include "Shapes.h"
#include <random>
using namespace std;

std::mt19937 ranGen;
void TestOne();		// Test 2D constructors
void TestTwo();		// Test 3D constructors
void TestThree();	// Test Scale
void TestFour();	// Largest/smallest area of 2D shapes
void TestFive();	// Largest/smallest area of 3D shapes
void RandomShapes(vector<Shape *> &shapes);
void Random2DShapes(vector<Shape2D *> &shapes);
void Random3DShapes(vector<Shape3D *> &shapes);
template<typename T>
void Cleanup(vector<T *> &shapes);
int RandomInt(int min, int max);
float RandomFloat(float min, float max);

#define rf RandomFloat(0, 10)

int main()
{
	cout << std::fixed << setprecision(2);
	
	// Input to seed a random number generator
	int randomInput;
	cin >> randomInput;
	ranGen.seed(randomInput);
	
	int testNum;
	cin >> testNum;

	if (testNum == 1)
		TestOne();
	else if (testNum == 2)
		TestTwo();
	else if (testNum == 3)
		TestThree();
	else if (testNum == 4)
		TestFour();
	else if (testNum == 5)
		TestFive();

	return 0;
}

// Testing constructors, default constructors and Display() for 2D shapes
void TestOne()
{
	// 2D shapes and their constructors
	vector<Shape *> shapes;
	shapes.push_back(new Square);
	shapes.push_back(new Square(1.1));
	shapes.push_back(new Triangle);
	shapes.push_back(new Triangle(4.4, 5.5));
	shapes.push_back(new Circle);
	shapes.push_back(new Circle(6.6));
	
	for (unsigned int i = 0; i < shapes.size(); i++)
	{
		shapes[i]->Display();
		cout << endl;
	}
	for (unsigned int i = 0; i < shapes.size(); i++)
		delete shapes[i];
}

// Testing constructors, default constructors and Display() for 3D shapes
void TestTwo()
{
	vector<Shape *> shapes;

	shapes.push_back(new TriangularPyramid);
	shapes.push_back(new TriangularPyramid(1.1, 2.2, 3.3));
	shapes.push_back(new Cylinder);
	shapes.push_back(new Cylinder(7.7, 8.8));
	shapes.push_back(new Sphere);
	shapes.push_back(new Sphere(9.9));

	for (unsigned int i = 0; i < shapes.size(); i++)
	{
		shapes[i]->Display();
		cout << endl;
	}
	for (unsigned int i = 0; i < shapes.size(); i++)
		delete shapes[i];
}

// Testing a the scaling of random shapes
void TestThree()
{
	vector<Shape *>shapes;
	RandomShapes(shapes);

	for (unsigned int i = 0; i < shapes.size(); i++)
	{
		shapes[i]->Display();
		cout << endl;
	}
	float scaleFactor;
	cin >> scaleFactor;

	cout << "Scaling..." << endl << endl;
	for (unsigned int i = 0; i < shapes.size(); i++)
	{
		shapes[i]->Scale(scaleFactor);
		shapes[i]->Display();
		cout << endl;
	}

	Cleanup(shapes);
}

// largest/smallest 2D shapes
void TestFour()
{
	vector<Shape2D *>shapes;
	Random2DShapes(shapes);
	for (unsigned int i = 0; i < shapes.size(); i++)
		shapes[i]->ShowArea();

	Shape2D *smallestShape = shapes[0];
	Shape2D *largestShape = shapes[0];

	// Find the largest and smallest shapes
	for (unsigned int i = 1; i < shapes.size(); i++)
	{
		if (*smallestShape > *shapes[i])
			smallestShape = shapes[i];
		if (*largestShape < *shapes[i])
			largestShape = shapes[i];
	}
	cout << endl << "***** The smallest shape *****" << endl;
	smallestShape->Display();

	cout << endl << "***** The largest shape *****" << endl;
	largestShape->Display();

	Cleanup(shapes);
}

// largest/smallest 3D shapes
void TestFive()
{
	vector<Shape3D *>shapes;
	Random3DShapes(shapes);

	Shape3D *smallestShape = shapes[0];
	Shape3D *largestShape = shapes[0];
	for (unsigned int i = 0; i < shapes.size(); i++)
		shapes[i]->ShowVolume();

	// Find the largest and smallest shapes
	for (unsigned int i = 0; i < shapes.size(); i++)
	{
		if (*smallestShape > *shapes[i])
			smallestShape = shapes[i];
		if (*largestShape < *shapes[i])
			largestShape = shapes[i];
	}
	cout << endl << "***** The smallest shape *****" << endl;
	smallestShape->Display();

	cout << endl << "***** The largest shape *****" << endl;
	largestShape->Display();

	Cleanup(shapes);
}

void RandomShapes(vector<Shape *> &shapes)
{
	int shapeCount = RandomInt(2, 6);
	for (int i = 0; i < shapeCount; i++)
	{
		int shapeType = RandomInt(0, 5);
		Shape *shape = nullptr;
		if (shapeType == 0) shape = new Square(rf);
		if (shapeType == 1) shape = new Triangle(rf, rf);
		if (shapeType == 2) shape = new Circle(rf);
		if (shapeType == 3) shape = new TriangularPyramid(rf, rf, rf);
		if (shapeType == 4) shape = new Cylinder(rf, rf);
		if (shapeType == 5) shape = new Sphere(rf);
		shapes.push_back(shape);
	}
}
void Random2DShapes(vector<Shape2D *> &shapes)
{
	int shapeCount = RandomInt(2, 6);
	for (int i = 0; i < shapeCount; i++)
	{
		int shapeType = RandomInt(0, 2);
		Shape2D *shape = nullptr;
		if (shapeType == 0) shape = new Square(rf);
		if (shapeType == 1) shape = new Triangle(rf, rf);
		if (shapeType == 2) shape = new Circle(rf);
		shapes.push_back(shape);
	}
}
void Random3DShapes(vector<Shape3D *> &shapes)
{
	int shapeCount = RandomInt(2, 6);
	for (int i = 0; i < shapeCount; i++)
	{
		int shapeType = RandomInt(0, 2);
		Shape3D *shape = nullptr;
		if (shapeType == 0) shape = new TriangularPyramid(rf, rf, rf);
		if (shapeType == 1) shape = new Cylinder(rf, rf);
		if (shapeType == 2) shape = new Sphere(rf);
		shapes.push_back(shape);
	}
}

template <typename T>
void Cleanup(vector<T *> &shapes)
{
	for (unsigned int i = 0; i < shapes.size(); i++)
		delete shapes[i];
}

int RandomInt(int min, int max)
{
	std::uniform_int_distribution<int> dist(min, max);
	return dist(ranGen);
}

float RandomFloat(float min, float max)
{
	std::uniform_real_distribution<> dist(min, max);
	return dist(ranGen);
}
