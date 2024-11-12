CREATE TABLE ToyCarOrders (
    OrderNumber INT NOT NULL,
    QuantityOrdered INT NOT NULL,
    PriceEach DEC(38, 2) NOT NULL,
    OrderLineNumber INT NOT NULL,
    Sales DEC(38, 2) NOT NULL,
    OrderDate DATE NOT NULL,
    DaysSinceLastOrder INT,
    ProductLine VARCHAR2(255),
    CustomerName VARCHAR2(255),
    AddressLine1 VARCHAR2(255),
    City VARCHAR2(255),
    PostalCode VARCHAR2(255),
    Country VARCHAR2(255),
    ContactLastName VARCHAR2(255),
    ContactFirstName VARCHAR2(255),
    DealSize VARCHAR2(255),
    PRIMARY KEY (OrderNumber, OrderLineNumber)
);