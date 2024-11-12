SELECT ROUND(AVG(Sales),2) AS AvgSales, MIN(Sales) AS MinSales, MAX(Sales) AS MaxSales,
SUM(Sales) As TotalSales
FROM ToyCarOrders
WHERE Country <> 'Germany';