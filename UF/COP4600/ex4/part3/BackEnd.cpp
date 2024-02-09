//Author: Zachery Utt

#include "BackEnd.h"
#include <algorithm>

//Dummy implementation of flightBookingQuery so you can check that your tests can compile

//NOTE: you should not be implementing these methods! You should be writing tests for them.



//flightBookingQuery
//The final implementation will return a vector<Booking> per the specifications

std::vector<Booking> flightBookingQuery(std::vector<Flight*>& allFlights, Airport& source, Airport& dest) {
	
	//This version always retuns an empty vector just so you can compile, link, and run your tests
	
	std::vector<Booking> toReturn;
	std::vector<Flight*> flights;
	for (Flight* flight : allFlights) {
		if (flight->getCapacity() < 1) {continue;}
		if (flight->getSource().getAirportCode() == source.getAirportCode() && flight->getDestination().getAirportCode() == dest.getAirportCode()) {
			flights.clear();
			flights.push_back(flight);
			toReturn.push_back(Booking(flights));
		}
	}
	if (toReturn.size() != 0) {
		std::sort(toReturn.begin(),
				  toReturn.end(),
				  [](Booking a, Booking b) { return a.getFlights()[0]->getTakeoffHour() < b.getFlights()[0]->getTakeoffHour(); });
		return toReturn;
	}

	for (Flight* flight : allFlights) {
		if (flight->getCapacity() < 1) {continue;}
		if (flight->getSource().getAirportCode() == source.getAirportCode()) {
			for (Flight* connection : allFlights) {
				if (connection->getCapacity() < 1) {continue;}
				if (connection->getDestination().getAirportCode() == dest.getAirportCode()
					&& connection->getTakeoffHour() >= 2 + flight->getTakeoffHour() 
					&& flight != connection
					&& flight->getDestination().getAirportCode() == connection->getSource().getAirportCode()) {
						flights.clear();
						flights.push_back(flight);
						flights.push_back(connection);
						toReturn.push_back(Booking(flights));
				}
			}
		}
	}
	if (toReturn.size() != 0) {
		std::sort(toReturn.begin(),
				  toReturn.end(),
				  [](Booking a, Booking b) { return std::min(a.getFlights()[0]->getTakeoffHour(), a.getFlights()[1]->getTakeoffHour()) < std::min(b.getFlights()[0]->getTakeoffHour(), b.getFlights()[1]->getTakeoffHour()); });
	}

	return toReturn;
}

//purchaseBooking
//The final implementation will process a Booking when it is purchased
void purchaseBooking(Booking& toPurchase) {
	//This version does nothing just so you can compile, link, and run your tests
	for (Flight* flight : toPurchase.getFlights()) {
		flight->decrementCapacity();
	}
}
