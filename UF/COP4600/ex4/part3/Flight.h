//Author: Zachery Utt

#pragma once

#include "Airport.h"

//A Flight is a scheduled flight from a source airport to a destination airport

class Flight {

private:
	int capacity;
	int takeoffTime;
	Airport& source;
	Airport& destination;

public:
	Flight(int capacity, int takeoffTime, Airport& source, Airport& destination) :
			capacity(capacity), takeoffTime(takeoffTime), source(source), destination(destination) { };
	
	int getTakeoffHour() {
		return this->takeoffTime;
	}


	int getCapacity() {
		return this->capacity;
	}

	void decrementCapacity() {
		this->capacity--;
	}

	Airport& getSource() {
		return this->source;	
	}
	
	Airport& getDestination() {
		return this->destination;	
	}
};
