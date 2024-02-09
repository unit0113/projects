//Author: Zachery Utt

#pragma once

#include <vector>
#include "Flight.h"

// A Booking object represents one or many flights (in the case of a connection)
// that satisfy the query

class Booking {

private:
	//Holds the Flight objects used for this trip
	std::vector<Flight*> flights;	

public:

	//Booking Constructor
	Booking(std::vector<Flight*>& flights) : flights(flights) {

	};

	//returns a copy of the flights member
	//which hold pointers to the Flight objects
	std::vector<Flight*> getFlights() {
		return flights;
	}
};

