//Author: Zachery Utt

#pragma once

#include "Booking.h"
#include "Flight.h"
#include "Airport.h"

#include <vector>

// the functions in this header are the  the Subjects Under Test!


// flightBookingQuery computes all available Booking objects per the specification
// it selects relevant flights from vector<Flight*> allFlights
// and returns Booking objects encapsulating the relevant flights
//
//	vector<Flight*> allFlights: list of ALL flights available
//	Airport source: reference to the source airport to query
//	Airport des:	reference to the destination airport to query

std::vector<Booking> flightBookingQuery(std::vector<Flight*>& allFlights, Airport& source, Airport& dest);


// purchaseBooking completes the purchase for a Booking selected by the user
void purchaseBooking(Booking& toPurchase);
