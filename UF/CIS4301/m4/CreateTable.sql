CREATE TABLE ToyCarOrders (
    OrderNumber INT NOT NULL,
    QuantityOrdered INT NOT NULL,
    PriceEach FLOAT NOT NULL,
    OrderLineNumber INT NOT NULL,
    Sales DOUBLE PRECISION,
    OrderDate DATE NOT NULL,
    DaysSinceLastOrder INT,
    ProductLine VARCHAR2(20),
    CustomerName VARCHAR2(40) NOT NULL,
    AddressLine1 VARCHAR2(50) NOT NULL,
    City VARCHAR2(20) NOT NULL,
    PostalCode VARCHAR2(15) NOT NULL,
    Country VARCHAR2(15) NOT NULL,
    ContactLastName VARCHAR2(20),
    ContactFirstName VARCHAR2(15),
    DealSize VARCHAR2(10),
    PRIMARY KEY (OrderNumber, OrderLineNumber)
);