SELECT OrderNumber, OrderLineNumber, QuantityOrdered, PriceEach, Sales
FROM ToyCarOrders
WHERE (OrderNumber = 10109 AND OrderLineNumber = 4)
OR (OrderNumber = 10100 AND OrderLineNumber = 2)
OR (OrderNumber = 10101 AND OrderLineNumber = 1);