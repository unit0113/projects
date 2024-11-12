DROP VIEW GDPResults;

CREATE VIEW GDPResults AS 
SELECT C.name AS country,
	RANK() OVER(ORDER BY E.gdp DESC) AS GDP_Rank,
	RANK() OVER(ORDER BY E.gdp / C.population DESC) AS GDPPC_Rank,
	RANK() OVER(ORDER BY E.industry DESC) AS IGDP_Rank,
	ROUND(E.gdp, 2) AS GDP_in_millions,
	ROUND(E.gdp * 1000000 / C.population, 2) AS GDPPC ,
	ROUND(E.gdp * E.industry, 2) AS IGDP_in_millions,
	C.population
FROM economy E, country C
WHERE E.country = C.code
    AND E.gdp IS NOT NULL
    AND E.industry IS NOT NULL
    AND C.population IS NOT NULL;


SELECT *
FROM GDPResults
ORDER BY GDPPC DESC;


DROP VIEW GDPOfCountry;

CREATE VIEW GDPOfCountry AS 
SELECT country, GDP_Rank, GDPPC_Rank, IGDP_Rank,
	LPAD(TO_CHAR(GDP_in_millions, '99999990D99'),12) AS GDP_in_millions, 
	LPAD(TO_CHAR(GDPPC, '99999990D99'),9) AS GDPPC,
	LPAD(TO_CHAR(IGDP_in_millions, '999999990D99'),12) AS IGDP_in_millions,
	population
FROM GDPResults
ORDER BY GDP_RANK;


SELECT *
FROM GDPOfCountry;



DROP VIEW GDPByIndsustryPercentage;

CREATE VIEW GDPByIndsustryPercentage AS 
SELECT country, IGDP_Rank, GDP_Rank, GDPPC_Rank, 
	LPAD(TO_CHAR(IGDP_in_millions, '999999990D99'),12) AS IGDP_in_millions,
	LPAD(TO_CHAR(GDP_in_millions, '99999990D99'),12) AS GDP_in_millions, 
	LPAD(TO_CHAR(GDPPC, '99999990D99'),9) AS GDPPC,
	population
FROM GDPResults
ORDER BY IGDP_Rank;

SELECT *
FROM GDPByIndsustryPercentage;



DROP VIEW GDPPerCapita;

CREATE VIEW GDPPerCapita AS
SELECT country, GDPPC_Rank, IGDP_Rank, GDP_Rank,
	LPAD(TO_CHAR(GDPPC, '99999990D99'),9) AS GDPPC,
	LPAD(TO_CHAR(IGDP_in_millions, '999999990D99'),12) AS IGDP_in_millions,
	LPAD(TO_CHAR(GDP_in_millions, '99999990D99'),12) AS GDP_in_millions, 
	population
FROM GDPResults
ORDER BY GDPPC_Rank;

SELECT *
FROM GDPPerCapita;
