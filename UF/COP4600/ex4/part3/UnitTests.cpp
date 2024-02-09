#include "gtest/gtest.h"
#include "BackEnd.h"


TEST(BookingCreationSuite, createDirect) {
	Airport src("MIA");
	Airport dest("OIA");
	Flight flight1(10, 6, src, dest);

	std::vector<Flight*> flights;
	flights.push_back(&flight1);
	std::vector<Booking> bookings = flightBookingQuery(flights, src, dest);

	// Ensure booking was created with the direct flight
	ASSERT_EQ(bookings.size(), 1);
	// Ensure booking has one flight
	ASSERT_EQ(bookings[0].getFlights().size(), 1);
	// Ensure the flight is the one provided
	ASSERT_EQ(bookings[0].getFlights()[0], &flight1);
	// Confirm flight data has not changed
	ASSERT_EQ(bookings[0].getFlights()[0]->getCapacity(), 10);
	ASSERT_EQ(bookings[0].getFlights()[0]->getTakeoffHour(), 6);
	ASSERT_EQ(bookings[0].getFlights()[0]->getSource().getAirportCode(), "MIA");
	ASSERT_EQ(bookings[0].getFlights()[0]->getDestination().getAirportCode(), "OIA");
}

TEST(BookingCreationSuite, createDirectConnectionOption) {
	Airport src("MIA");
	Airport dest("OIA");
	Airport conn("TAL");
	Flight flight1(10, 6, src, dest);
	Flight conn1(10, 6, src, conn);
	Flight conn2(10, 10, conn, dest);

	std::vector<Flight*> flights;
	flights.push_back(&flight1);
	flights.push_back(&conn1);
	flights.push_back(&conn2);
	std::vector<Booking> bookings = flightBookingQuery(flights, src, dest);

	// Assert only the direct flight was included
	ASSERT_EQ(bookings.size(), 1);
	// Ensure booking has one flight
	ASSERT_EQ(bookings[0].getFlights().size(), 1);
	// Assert that the only flight is the direct flight
	ASSERT_EQ(bookings[0].getFlights()[0], &flight1);
}

TEST(BookingCreationSuite, singleConnectionOptions) {
	Airport src("MIA");
	Airport dest("OIA");
	Airport conn1("TAL");
	Flight conn11(10, 6, src, conn1);
	Flight conn12(10, 10, conn1, dest);

	std::vector<Flight*> flights;
	flights.push_back(&conn11);
	flights.push_back(&conn12);
	std::vector<Booking> bookings = flightBookingQuery(flights, src, dest);

	// Assert that the connection pair created a booking
	ASSERT_EQ(bookings.size(), 1);
	// Assert that the booking contains two flights
	ASSERT_EQ(bookings[0].getFlights().size(), 2);
	// Assert that the flights are not the same
	ASSERT_NE(bookings[0].getFlights()[0], bookings[0].getFlights()[1]);
	// Assert that the flights are the provided flights
	ASSERT_TRUE((bookings[0].getFlights()[0] == &conn11) || (bookings[0].getFlights()[0] == &conn12));
	ASSERT_TRUE((bookings[0].getFlights()[1] == &conn11) || (bookings[0].getFlights()[1] == &conn12));
	// Assert that flight data has not changed
	if (bookings[0].getFlights()[0] == flights[0]) {
		// Flight 1
		ASSERT_EQ(bookings[0].getFlights()[0]->getCapacity(), 10);
		ASSERT_EQ(bookings[0].getFlights()[0]->getTakeoffHour(), 6);
		ASSERT_EQ(bookings[0].getFlights()[0]->getSource().getAirportCode(), "MIA");
		ASSERT_EQ(bookings[0].getFlights()[0]->getDestination().getAirportCode(), "TAL");

		// Flight 2
		ASSERT_EQ(bookings[0].getFlights()[1]->getCapacity(), 10);
		ASSERT_EQ(bookings[0].getFlights()[1]->getTakeoffHour(), 10);
		ASSERT_EQ(bookings[0].getFlights()[1]->getSource().getAirportCode(), "TAL");
		ASSERT_EQ(bookings[0].getFlights()[1]->getDestination().getAirportCode(), "OIA");
	}
	else {
		// Flight 1
		ASSERT_EQ(bookings[0].getFlights()[1]->getCapacity(), 10);
		ASSERT_EQ(bookings[0].getFlights()[1]->getTakeoffHour(), 6);
		ASSERT_EQ(bookings[0].getFlights()[1]->getSource().getAirportCode(), "MIA");
		ASSERT_EQ(bookings[0].getFlights()[1]->getDestination().getAirportCode(), "TAL");

		// Flight 2
		ASSERT_EQ(bookings[0].getFlights()[0]->getCapacity(), 10);
		ASSERT_EQ(bookings[0].getFlights()[0]->getTakeoffHour(), 10);
		ASSERT_EQ(bookings[0].getFlights()[0]->getSource().getAirportCode(), "TAL");
		ASSERT_EQ(bookings[0].getFlights()[0]->getDestination().getAirportCode(), "OIA");
	}
}

TEST(BookingCreationSuite, multipleConnectionOptions) {
	Airport src("MIA");
	Airport dest("OIA");
	Airport conn1("TAL");
	Airport conn2("CCB");
	Flight conn11(10, 6, src, conn1);
	Flight conn12(10, 10, conn1, dest);
	Flight conn21(10, 6, src, conn2);
	Flight conn22(10, 10, conn2, dest);

	std::vector<Flight*> flights;
	flights.push_back(&conn11);
	flights.push_back(&conn12);
	flights.push_back(&conn21);
	flights.push_back(&conn22);
	std::vector<Booking> bookings = flightBookingQuery(flights, src, dest);

	// Assert that both connection paths are included
	ASSERT_EQ(bookings.size(), 2);
	// Ensure that each booking has two flights
	ASSERT_EQ(bookings[0].getFlights().size(), 2);
	ASSERT_EQ(bookings[1].getFlights().size(), 2);
	// Assert that each connection path in valid (also checks ordering)
	ASSERT_TRUE((bookings[0].getFlights()[0] == flights[0]) || (bookings[0].getFlights()[1] == flights[0]));
	ASSERT_TRUE((bookings[0].getFlights()[0] == flights[1]) || (bookings[0].getFlights()[1] == flights[1]));
	ASSERT_TRUE((bookings[1].getFlights()[0] == flights[2]) || (bookings[1].getFlights()[1] == flights[2]));
	ASSERT_TRUE((bookings[1].getFlights()[0] == flights[3]) || (bookings[1].getFlights()[1] == flights[3]));
	// Assert that the flights in each booking are not the same
	ASSERT_NE(bookings[0].getFlights()[0], bookings[0].getFlights()[1]);
	ASSERT_NE(bookings[1].getFlights()[0], bookings[1].getFlights()[1]);
}

TEST(BookingCreationSuite, noFullFlights) {
	// Init objects
	Airport src("MIA");
	Airport dest("OIA");
	Airport conn1("TAL");
	Flight flight1(0, 6, src, dest);
	Flight flight2(0, 8, src, dest);
	Flight conn11(0, 6, src, conn1);
	Flight conn12(0, 10, conn1, dest);
	Flight neg1(-1, 6, src, dest);
	Flight neg2(-1000, 6, src, dest);

	// Assemble flights
	std::vector<Flight*> flights;
	flights.push_back(&flight1);
	flights.push_back(&flight2);
	flights.push_back(&conn11);
	flights.push_back(&conn12);
	flights.push_back(&neg1);
	flights.push_back(&neg2);

	std::vector<Booking> bookings = flightBookingQuery(flights, src, dest);

	// Assert that full flights are excluded
	ASSERT_EQ(bookings.size(), 0);
}

TEST(BookingCreationSuite, bookingOrderingDirect) {
	Airport src("MIA");
	Airport dest("OIA");
	Flight flight1(10, 6, src, dest);
	Flight flight2(10, 8, src, dest);
	Flight flight3(10, 10, src, dest);
	Flight flight4(10, 0, src, dest);
	Flight flight5(10, 23, src, dest);

	std::vector<Flight*> flights;
	flights.push_back(&flight1);
	flights.push_back(&flight2);
	flights.push_back(&flight3);
	flights.push_back(&flight4);
	flights.push_back(&flight5);
	std::vector<Booking> bookings = flightBookingQuery(flights, src, dest);

	// Ensure booking was created with all flights
	ASSERT_EQ(bookings.size(), 5);
	// Ensure each booking has one flight
	ASSERT_EQ(bookings[0].getFlights().size(), 1);
	ASSERT_EQ(bookings[1].getFlights().size(), 1);
	ASSERT_EQ(bookings[2].getFlights().size(), 1);
	ASSERT_EQ(bookings[3].getFlights().size(), 1);
	ASSERT_EQ(bookings[4].getFlights().size(), 1);
	// Ensure the flights are provided in order of departure time
	ASSERT_EQ(bookings[0].getFlights()[0], &flight4);
	ASSERT_EQ(bookings[1].getFlights()[0], &flight1);
	ASSERT_EQ(bookings[2].getFlights()[0], &flight2);
	ASSERT_EQ(bookings[3].getFlights()[0], &flight3);
	ASSERT_EQ(bookings[4].getFlights()[0], &flight5);
}

TEST(BookingCreationSuite, bookingOrderingConnections) {
	Airport src("MIA");
	Airport dest("OIA");
	Airport conn1("TAL");
	Airport conn2("CCB");
	Flight conn11(10, 6, src, conn1);
	Flight conn12(10, 14, conn1, dest);
	Flight conn21(10, 8, src, conn2);
	Flight conn22(10, 12, conn2, dest);

	std::vector<Flight*> flights;
	flights.push_back(&conn11);
	flights.push_back(&conn12);
	flights.push_back(&conn21);
	flights.push_back(&conn22);
	std::vector<Booking> bookings = flightBookingQuery(flights, src, dest);

	// Assert both valid bookings are found
	ASSERT_EQ(bookings.size(), 2);
	// Asset each booking has two flights
	ASSERT_EQ(bookings[0].getFlights().size(), 2);
	ASSERT_EQ(bookings[1].getFlights().size(), 2);
	// Assert TAL booking is first
	ASSERT_TRUE((bookings[0].getFlights()[0] == &conn11) || (bookings[0].getFlights()[1] == &conn11));
	// Assert CCB booking is second
	ASSERT_TRUE((bookings[1].getFlights()[0] == &conn21) || (bookings[1].getFlights()[1] == &conn21));

	// Reverse input order of flights and repeat
	flights.clear();
	flights.push_back(&conn22);
	flights.push_back(&conn21);
	flights.push_back(&conn12);
	flights.push_back(&conn11);

	bookings = flightBookingQuery(flights, src, dest);

	// Assert both valid bookings are found
	ASSERT_EQ(bookings.size(), 2);
	// Asset each booking has two flights
	ASSERT_EQ(bookings[0].getFlights().size(), 2);
	ASSERT_EQ(bookings[1].getFlights().size(), 2);
	// Assert TAL booking is first
	ASSERT_TRUE((bookings[0].getFlights()[0] == &conn11) || (bookings[0].getFlights()[1] == &conn11));
	// Assert CCB booking is second
	ASSERT_TRUE((bookings[1].getFlights()[0] == &conn21) || (bookings[1].getFlights()[1] == &conn21));

}

TEST(BookingCreationSuite, onlyOneConnection) {
	Airport src("MIA");
	Airport dest("OIA");
	Airport con1("TAL");
	Airport con2("CCB");
	Airport con3("OMA");
	Flight conn1(10, 4, src, con1);
	Flight conn2(10, 8, con1, con2);
	Flight conn3(10, 12, con2, dest);
	Flight conn4(10, 16, con2, con3);
	Flight conn5(10, 16, con3, dest);

	std::vector<Flight*> flights;
	flights.push_back(&conn1);
	flights.push_back(&conn2);
	flights.push_back(&conn3);
	flights.push_back(&conn4);
	flights.push_back(&conn5);

	std::vector<Booking> bookings = flightBookingQuery(flights, src, dest);

	// Assert no valid booking
	ASSERT_EQ(bookings.size(), 0);
}

TEST(BookingCreationSuite, ensureValidDepartureSpacing) {
	Airport src("MIA");
	Airport dest("OIA");
	Airport con1("TAL");
	Flight conn1(10, 4, src, con1);
	Flight conn3(10, 5, con1, dest);
	Flight conn2(10, 1, con1, dest);

	std::vector<Flight*> flights;
	flights.push_back(&conn1);
	flights.push_back(&conn2);
	flights.push_back(&conn3);

	std::vector<Booking> bookings = flightBookingQuery(flights, src, dest);

	// Assert no valid booking
	ASSERT_EQ(bookings.size(), 0);

	Flight conn4(10, 6, con1, dest);
	flights.push_back(&conn4);
	bookings.clear();
	bookings = flightBookingQuery(flights, src, dest);

	// Assert new flight yields valid booking
	ASSERT_EQ(bookings.size(), 1);
	// Assert new booking does not include invalid flight
	ASSERT_TRUE((bookings[0].getFlights()[0] == flights[0]) || (bookings[0].getFlights()[0] == flights[3]));
	ASSERT_TRUE((bookings[0].getFlights()[1] == flights[0]) || (bookings[0].getFlights()[1] == flights[3]));
}


TEST(PurchaseBookingSuite, purchaseDirect) {
	Airport src("MIA");
	Airport dest("OIA");
	Flight flight1(10, 6, src, dest);

	std::vector<Flight*> flights;
	flights.push_back(&flight1);
	std::vector<Booking> bookings = flightBookingQuery(flights, src, dest);

	// Ensure flight retains capacity prior to purchase
	ASSERT_EQ(flight1.getCapacity(), 10);

	purchaseBooking(bookings[0]);

	// Assert that capacity has decreased
	ASSERT_EQ(flight1.getCapacity(), 9);
	// Ensure the flight is still the one provided
	ASSERT_EQ(bookings[0].getFlights(), flights);
	// Confirm other flight data has not changed
	ASSERT_EQ(bookings[0].getFlights()[0]->getTakeoffHour(), 6);
	ASSERT_EQ(bookings[0].getFlights()[0]->getSource().getAirportCode(), "MIA");
	ASSERT_EQ(bookings[0].getFlights()[0]->getDestination().getAirportCode(), "OIA");
}

TEST(PurchaseBookingSuite, purchaseConnection) {
	Airport src("MIA");
	Airport dest("OIA");
	Airport conn1("TAL");
	Flight conn11(10, 6, src, conn1);
	Flight conn12(10, 10, conn1, dest);

	std::vector<Flight*> flights;
	flights.push_back(&conn11);
	flights.push_back(&conn12);
	std::vector<Booking> bookings = flightBookingQuery(flights, src, dest);

	// Ensure flight retains capacity prior to purchase
	ASSERT_EQ(conn11.getCapacity(), 10);
	ASSERT_EQ(conn12.getCapacity(), 10);

	purchaseBooking(bookings[0]);

	// Assert that capacity has decreased
	ASSERT_EQ(conn11.getCapacity(), 9);
	ASSERT_EQ(conn12.getCapacity(), 9);
	// Assert that the connection pair created a booking
	ASSERT_EQ(bookings.size(), 1);
	// Assert that the booking contains two flights
	ASSERT_EQ(bookings[0].getFlights().size(), 2);
	// Assert that the flights are not the same
	ASSERT_NE(bookings[0].getFlights()[0], bookings[0].getFlights()[1]);
	// Assert that the flights are still the provided flights
	ASSERT_TRUE((bookings[0].getFlights()[0] == flights[0]) || (bookings[0].getFlights()[0] == flights[1]));
	ASSERT_TRUE((bookings[0].getFlights()[1] == flights[0]) || (bookings[0].getFlights()[1] == flights[1]));
	// Assert that other flight data still has not changed
	if (bookings[0].getFlights()[0] == flights[0]) {
		// Flight 1
		ASSERT_EQ(bookings[0].getFlights()[0]->getTakeoffHour(), 6);
		ASSERT_EQ(bookings[0].getFlights()[0]->getSource().getAirportCode(), "MIA");
		ASSERT_EQ(bookings[0].getFlights()[0]->getDestination().getAirportCode(), "TAL");

		// Flight 2
		ASSERT_EQ(bookings[0].getFlights()[1]->getTakeoffHour(), 10);
		ASSERT_EQ(bookings[0].getFlights()[1]->getSource().getAirportCode(), "TAL");
		ASSERT_EQ(bookings[0].getFlights()[1]->getDestination().getAirportCode(), "OIA");
	}
	else {
		// Flight 1
		ASSERT_EQ(bookings[0].getFlights()[1]->getTakeoffHour(), 6);
		ASSERT_EQ(bookings[0].getFlights()[1]->getSource().getAirportCode(), "MIA");
		ASSERT_EQ(bookings[0].getFlights()[1]->getDestination().getAirportCode(), "TAL");

		// Flight 2
		ASSERT_EQ(bookings[0].getFlights()[0]->getTakeoffHour(), 10);
		ASSERT_EQ(bookings[0].getFlights()[0]->getSource().getAirportCode(), "TAL");
		ASSERT_EQ(bookings[0].getFlights()[0]->getDestination().getAirportCode(), "OIA");
	}
}

TEST(PurchaseBookingSuite, decrementCapacityToZero) {
	Airport src("MIA");
	Airport dest("OIA");
	Flight flight1(1, 6, src, dest);

	std::vector<Flight*> flights;
	flights.push_back(&flight1);
	std::vector<Booking> bookings = flightBookingQuery(flights, src, dest);

	// Ensure flight retains capacity prior to purchase
	ASSERT_EQ(flight1.getCapacity(), 1);

	purchaseBooking(bookings[0]);

	// Assert that capacity has decreased to zero
	ASSERT_EQ(flight1.getCapacity(), 0);
}

TEST(PurchaseBookingSuite, repeatBookings) {
	Airport src("MIA");
	Airport dest("OIA");
	Flight flight1(2, 6, src, dest);

	std::vector<Flight*> flights;
	flights.push_back(&flight1);
	std::vector<Booking> bookings = flightBookingQuery(flights, src, dest);

	// Ensure flight retains capacity prior to purchase
	ASSERT_EQ(bookings.size(), 1);
	ASSERT_EQ(flight1.getCapacity(), 2);

	purchaseBooking(bookings[0]);

	// Assert that capacity has decreased
	ASSERT_EQ(flight1.getCapacity(), 1);

	bookings = flightBookingQuery(flights, src, dest);

	// Assert that flight is still valid booking
	ASSERT_EQ(bookings.size(), 1);

	purchaseBooking(bookings[0]);

	// Assert that capacity has decreased to zero
	ASSERT_EQ(flight1.getCapacity(), 0);

	// Assert that repeat of booking query produces no valid results
	bookings = flightBookingQuery(flights, src, dest);
	ASSERT_EQ(bookings.size(), 0);
}

TEST(PurchaseBookingSuite, repeatBookingsConnection) {
	Airport src("MIA");
	Airport dest("OIA");
	Airport conn1("TAL");
	Flight conn11(3, 6, src, conn1);
	Flight conn12(2, 10, conn1, dest);

	std::vector<Flight*> flights;
	flights.push_back(&conn11);
	flights.push_back(&conn12);
	std::vector<Booking> bookings = flightBookingQuery(flights, src, dest);

	// Ensure flight retains capacity prior to purchase
	ASSERT_EQ(bookings.size(), 1);
	ASSERT_EQ(conn11.getCapacity(), 3);
	ASSERT_EQ(conn12.getCapacity(), 2);

	purchaseBooking(bookings[0]);

	// Assert that capacity has decreased
	ASSERT_EQ(conn11.getCapacity(), 2);
	ASSERT_EQ(conn12.getCapacity(), 1);

	bookings = flightBookingQuery(flights, src, dest);

	// Assert that connection is still valid
	ASSERT_EQ(bookings.size(), 1);

	purchaseBooking(bookings[0]);

	// Assert that capacity has decreased
	ASSERT_EQ(conn11.getCapacity(), 1);
	ASSERT_EQ(conn12.getCapacity(), 0);

	// Assert that repeat of booking query produces no valid results
	bookings = flightBookingQuery(flights, src, dest);
	ASSERT_EQ(bookings.size(), 0);
}