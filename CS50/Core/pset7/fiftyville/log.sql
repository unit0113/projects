-- Keep a log of any SQL queries you execute as you solve the mystery.

-- Search crime scene reports by date and street
SELECT description FROM crime_scene_reports WHERE street = "Chamberlin Street" AND year = 2020 AND month = 7 AND day = 28;
--Theft of the CS50 duck took place at 10:15am at the Chamberlin Street courthouse. Interviews were conducted today with three witnesses who were present at the time — each of their interview transcripts mentions the courthouse.

--Search interviews by date and mentions of courthouse
SELECT name, transcript FROM interviews WHERE transcript LIKE "%courthouse%" AND month = 7 AND day = 28;
--Ruth | Sometime within ten minutes of the theft, I saw the thief get into a car in the courthouse parking lot and drive away. If you have security footage from the courthouse parking lot, you might want to look for cars that left the parking lot in that time frame.
--Eugene | I don't know the thief's name, but it was someone I recognized. Earlier this morning, before I arrived at the courthouse, I was walking by the ATM on Fifer Street and saw the thief there withdrawing some money.
--Raymond | As the thief was leaving the courthouse, they called someone who talked to them for less than a minute. In the call, I heard the thief say that they were planning to take the earliest flight out of Fiftyville tomorrow. The thief then asked the person on the other end of the phone to purchase the flight ticket.

--Three clues:
--between 10:15 and 10:25 thief got into a car: check security footage
--prior to 10:15 thief withdrew money from ATM on Fifer Street
--Phone call to accomplice for less than 1 min, earliest flight out on 29th, accomplice bought ticket (proabably?)

--Clue one: check courthouse security cams for time period mentioned earlier
SELECT activity, license_plate FROM courthouse_security_logs WHERE month = 7 AND day = 28 AND hour = 10 AND minute BETWEEN 15 AND 25;
--exit | 5P2BI95
--exit | 94KL13X
--exit | 6P58WS2
--exit | 4328GD8
--exit | G412CB7
--exit | L93JTIZ
--exit | 322W7JE
--exit | 0NTHK55
--Some possibilites for later

--Clue two: check ATM records, join with bank accounts and people to get names
SELECT name, amount, phone_number, passport_number, license_plate FROM atm_transactions JOIN bank_accounts ON atm_transactions.account_number = bank_accounts.account_number JOIN people ON bank_accounts.person_id = people.id WHERE month = 7 AND day = 28 AND transaction_type = "withdraw" AND atm_location = "Fifer Street";
--Ernest | 50 | (367) 555-5533 | 5773159633 | 94KL13X
--Russell | 35 | (770) 555-1861 | 3592750733 | 322W7JE
--Roy | 80 | (122) 555-4581 | 4408372428 | QX4YZN3
--Bobby | 20 | (826) 555-1652 | 9878712108 | 30G67EN
--Elizabeth | 20 | (829) 555-5269 | 7049073643 | L93JTIZ
--Danielle | 48 | (389) 555-5198 | 8496433585 | 4328GD8
--Madison | 60 | (286) 555-6063 | 1988161715 | 1106N58
--Victoria | 30 | (338) 555-6650 | 9586786673 | 8X428L0

--Combine two previous queries to narrow down suspects
SELECT name, phone_number, passport_number, people.license_plate FROM atm_transactions JOIN bank_accounts ON atm_transactions.account_number = bank_accounts.account_number JOIN people ON bank_accounts.person_id = people.id JOIN courthouse_security_logs ON courthouse_security_logs.license_plate = people.license_plate WHERE atm_transactions.month = 7 AND atm_transactions.day = 28 AND courthouse_security_logs.month = 7 AND courthouse_security_logs.day = 28 AND transaction_type = "withdraw" AND atm_location = "Fifer Street" AND courthouse_security_logs.hour = 10 AND courthouse_security_logs.minute BETWEEN 15 AND 25;
--Ernest | (367) 555-5533 | 5773159633 | 94KL13X
--Russell | (770) 555-1861 | 3592750733 | 322W7JE
--Elizabeth | (829) 555-5269 | 7049073643 | L93JTIZ
--Danielle | (389) 555-5198 | 8496433585 | 4328GD8
--And then there where four... The plot, like my waistline, thickens...

--Clue three: phone calls, cross check time leaving courthouse (10:15-10:25)(note: apparently can't do), duration (sub 1 min)
SELECT name, phone_number FROM phone_calls JOIN people ON phone_calls.caller = people.phone_number WHERE month = 7 AND day = 28 AND duration < 60;
--Roger | (130) 555-0289
--Evelyn | (499) 555-9472
--Ernest | (367) 555-5533
--Evelyn | (499) 555-9472
--Madison | (286) 555-6063
--Russell | (770) 555-1861
--Kimberly | (031) 555-6622
--Bobby | (826) 555-1652
--Victoria | (338) 555-6650

--Combine previous two queries
SELECT name, phone_number, passport_number, people.license_plate FROM phone_calls JOIN people ON phone_calls.caller = people.phone_number WHERE month = 7 AND day = 28 AND duration < 60 AND phone_number IN (SELECT phone_number FROM atm_transactions JOIN bank_accounts ON atm_transactions.account_number = bank_accounts.account_number JOIN people ON bank_accounts.person_id = people.id JOIN courthouse_security_logs ON courthouse_security_logs.license_plate = people.license_plate WHERE atm_transactions.month = 7 AND atm_transactions.day = 28 AND courthouse_security_logs.month = 7 AND courthouse_security_logs.day = 28 AND transaction_type = "withdraw" AND atm_location = "Fifer Street" AND courthouse_security_logs.hour = 10 AND courthouse_security_logs.minute BETWEEN 15 AND 25);
--Ernest | (367) 555-5533 | 5773159633 | 94KL13X
--Russell | (770) 555-1861 | 3592750733 | 322W7JE

--Get the people these two called
SELECT distinct(name), phone_number, account_number, caller FROM people JOIN bank_accounts ON bank_accounts.person_id = people.id JOIN phone_calls ON phone_calls.caller = people.phone_number WHERE phone_number IN (SELECT receiver FROM phone_calls WHERE caller IN (SELECT phone_number FROM phone_calls JOIN people ON phone_calls.caller = people.phone_number WHERE month = 7 AND day = 28 AND duration < 60 AND phone_number IN (SELECT phone_number FROM atm_transactions JOIN bank_accounts ON atm_transactions.account_number = bank_accounts.account_number JOIN people ON bank_accounts.person_id = people.id JOIN courthouse_security_logs ON courthouse_security_logs.license_plate = people.license_plate WHERE atm_transactions.month = 7 AND atm_transactions.day = 28 AND courthouse_security_logs.month = 7 AND courthouse_security_logs.day = 28 AND transaction_type = "withdraw" AND atm_location = "Fifer Street" AND courthouse_security_logs.hour = 10 AND courthouse_security_logs.minute BETWEEN 15 AND 25)));
--Karen | (841) 555-3728 | 32212020 | (841) 555-3728
--Philip | (725) 555-3243 | 47746428 | (725) 555-3243
--Tyler | (660) 555-3095 | 44432923 | (660) 555-3095
--Nicole | (123) 555-5144 | 83997425 | (123) 555-5144
--Pamela | (113) 555-7544 | 16654966 | (113) 555-7544
--Joseph | (238) 555-5554 | 79806482 | (238) 555-5554
--Berthold | (375) 555-8161 | 94751264 | (375) 555-8161
--Charlotte | (455) 555-5315 | 15871517 | (455) 555-5315

--Queries getting long, new avenue, find passengers on earliest flight on 29th
SELECT name FROM people JOIN passengers ON passengers.passport_number = people.passport_number WHERE flight_id IN (SELECT id FROM flights WHERE origin_airport_id IN (SELECT id FROM airports WHERE city = "Fiftyville") AND month = 7 AND day = 29 ORDER BY hour ASC, minute ASC LIMIT 1);
--Doris
--Roger
--Ernest
--Edward
--Evelyn
--Madison
--Bobby
--Danielle

--Combine previous query with the ultimate super-long one (that had two results)
SELECT name FROM phone_calls JOIN people ON phone_calls.caller = people.phone_number WHERE month = 7 AND day = 28 AND duration < 60 AND phone_number IN (SELECT phone_number FROM atm_transactions JOIN bank_accounts ON atm_transactions.account_number = bank_accounts.account_number JOIN people ON bank_accounts.person_id = people.id JOIN courthouse_security_logs ON courthouse_security_logs.license_plate = people.license_plate WHERE atm_transactions.month = 7 AND atm_transactions.day = 28 AND courthouse_security_logs.month = 7 AND courthouse_security_logs.day = 28 AND transaction_type = "withdraw" AND atm_location = "Fifer Street" AND courthouse_security_logs.hour = 10 AND courthouse_security_logs.minute BETWEEN 15 AND 25) AND name IN (SELECT name FROM people JOIN passengers ON passengers.passport_number = people.passport_number WHERE flight_id IN (SELECT id FROM flights WHERE origin_airport_id IN (SELECT id FROM airports WHERE city = "Fiftyville") AND month = 7 AND day = 29 ORDER BY hour ASC, minute ASC LIMIT 1));
--Ernest
--Damn you Ernest, I trusted you...

--Get where he flew to
SELECT city FROM airports WHERE id IN (SELECT destination_airport_id FROM flights WHERE month = 7 AND day = 29 AND id IN (SELECT flight_id FROM passengers JOIN people ON passengers.passport_number = people.passport_number WHERE name = "Ernest"));
--London

--Find accomplice time, see who Ernest called on the 28th
SELECT name FROM people WHERE phone_number IN (SELECT receiver FROM phone_calls WHERE year = 2020 AND month = 7 AND day = 28 AND duration < 60 AND caller IN (SELECT phone_number FROM people WHERE name = "Ernest"));
--Berthold

--First try!