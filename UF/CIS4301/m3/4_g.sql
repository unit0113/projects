SELECT QuantityOrdered, PriceEach, Sales, QuantityOrdered * PriceEach AS "Computed Value",
QuantityOrdered * PriceEach - Sales AS Difference
FROM ToyCarOrders;