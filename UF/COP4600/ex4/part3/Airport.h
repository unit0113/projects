#pragma once
//Author: Zachery Utt


#include <string>

//An object representing an airport

class Airport {
private:
	std::string airportCode;
	
public:
	// Constructor for Airport
	// airportCode is a three letter string airport code
	Airport(const std::string& airportCode) {
		this->airportCode = airportCode;
	}
	
	//returns a copy of the airport code
	std::string getAirportCode() {
		return this->airportCode;
	}

};
