--1
SELECT continent, SUM(percentage * population / 100) AS totalPopulation
FROM encompasses, country
WHERE encompasses.country = country.code
GROUP BY continent
ORDER BY totalPopulation DESC;

--2
SELECT C.name AS country, E1.country AS countrycode, E1.continent AS continent1, E2.continent AS continent2,
    (SELECT COUNT(*)
    FROM city, country
    WHERE city.country = country.code and country.code = E1.country) AS NumberOfCities
FROM encompasses E1, encompasses E2, country C
WHERE E1.country = E2.country
    AND E1.continent <> E2.continent
    AND E1.continent < E2.continent
    AND E1.country = C.code
ORDER BY continent1, continent2, country;
    
--3
SELECT name AS continent, ROUND(totalPopulation / area, 2) AS avgPopulationDensity,
    ROUND(area / totalPopulation, 4) AS avgArealConcentration
FROM continent C, (SELECT continent, SUM(percentage * population / 100) AS totalPopulation
    FROM encompasses, country
    WHERE encompasses.country = country.code
    GROUP BY continent
    ORDER BY totalPopulation DESC) P
WHERE C.name = P.continent
ORDER BY avgPopulationDensity DESC;

--4
SELECT C.name AS country, ROUND(AVG(elevation),2) AS AvgElevation,
    (SELECT COUNT(*)
    FROM airport A, country C2
    WHERE A.country = C2.code AND C2.name = C.name) AS NumAirports
FROM airport A, country C
WHERE A.country = C.code
GROUP BY C.name
ORDER BY AvgElevation DESC
FETCH FIRST 5 ROWS ONLY;

--5
WITH borderecon AS (SELECT country1, E1.gdp as gdp1, country2, E2.gdp AS gdp2
FROM borders B, economy E1, economy E2
WHERE country2 = E2.country AND country1 = E1.country),

bigborderecon AS(SELECT country1 AS country, gdp1 AS countrygdp, country2 AS bordercountry, gdp2 AS bordergdp
FROM borderecon
UNION
SELECT country2 AS country, gdp2 AS countrygdp, country1 AS bordercountry, gdp1 AS bordergdp
FROM borderecon),

filteredBBE AS (Select * FROM bigborderecon WHERE countrygdp < bordergdp)

SELECT DISTINCT FBBE1.country, MIN(FBBE1.countrygdp) AS countrygdp,
    (SELECT COUNT(*) FROM (SELECT DISTINCT FBBE4.bordercountry FROM filteredBBE FBBE4 WHERE FBBE1.country = FBBE4.country)) AS numWealthier,
    (SELECT ROUND(AVG(bordergdp), 2) FROM (SELECT DISTINCT FBBE4.bordergdp FROM filteredBBE FBBE4 WHERE FBBE1.country = FBBE4.country)) AS avgBorderingGDP
FROM filteredBBE FBBE1, filteredBBE FBBE2, filteredBBE FBBE3
WHERE FBBE1.bordercountry <> FBBE2.bordercountry 
    AND FBBE2.bordercountry <> FBBE3.bordercountry 
    AND FBBE1.bordercountry <> FBBE3.bordercountry 
    AND FBBE1.country = FBBE2.country 
    AND FBBE2.country = FBBE3.country 
    AND FBBE1.country = FBBE3.country 
GROUP BY FBBE1.country
ORDER BY numWealthier DESC;



--6
WITH cpops AS (SELECT C.code, C.name, MAX(P.population) AS population
    FROM country C, countrypops P
    WHERE C.code = P.country
        AND P.population > 50000000
    GROUP BY C.code, C.name
    ORDER BY population DESC),

countrycapital AS (SELECT C.name, C.capital, C2.population as capitalpop
    FROM Country C, City C2
    WHERE C.capital = C2.name
        AND C2.population > 1000000
    ORDER BY capitalpop DESC)

SELECT C1.name, C2.capital, C2.capitalpop, C1.population, E.gdp,
    ROUND(C2.capitalpop / C1.population, 3) AS captialpoppercentage
FROM cpops C1, countrycapital C2, economy E
WHERE C1.name = c2.name
    AND C1.code = E.country
    AND E.gdp > 100000
ORDER BY captialpoppercentage DESC 
FETCH FIRST 10 ROWS ONLY;

--7
WITH commonlanguages AS (SELECT *
FROM language
WHERE percentage >= 10),

multilingualcountries AS (SELECT DISTINCT CL1.country
    FROM commonlanguages CL1, commonlanguages CL2, commonlanguages CL3
    WHERE CL1.country = CL2.country
        AND CL2.country = CL3.country
        AND CL1.country = CL3.country
        AND CL1.name <> CL2.name
        AND CL2.name <> CL3.name
        AND CL1.name <> CL3.name)
    
SELECT M.country, L.name, L.percentage
FROM multilingualcountries M, language L
WHERE M.country = L.country
    AND L.percentage >= 10
ORDER BY M.country, L.percentage DESC;

--8
WITH tallMountains AS (SELECT GM.country, M.name, MAX(M.elevation) AS maxElevation
    FROM geo_mountain GM, mountain M
    WHERE GM.mountain = M.name AND M.elevation > 5000
    GROUP BY GM.country, M.name),

mountainCount AS (SELECT country, COUNT(name) AS numMountains
    FROM tallMountains TM
    GROUP BY country),

mountainInfo AS (SELECT TM.country, MAX(TM.maxElevation) AS maxElevation, MAX(MC.numMountains) AS numMountains
    FROM tallMountains TM, mountainCount MC
    WHERE TM.country = MC.country
    GROUP BY TM.country),

longRivers AS (SELECT GR.country, R.name, MAX(R.length) AS maxLength
    FROM river R, geo_river GR
    WHERE R.name = GR.river AND R.length > 3000
    GROUP BY GR.country, R.name),

riverCount AS (SELECT country, COUNT(name) AS numRivers
    FROM longRivers LR
    GROUP BY country),

riverInfo AS (SELECT LR.country, MAX(LR.maxlength) AS maxLength, MAX(RC.numrivers) AS numRivers
    FROM longRivers LR, riverCount RC
    WHERE LR.country = RC.country
    GROUP BY LR.country)
    
SELECT C.name, MI.maxElevation, RI.maxLength, MI.numMountains, RI.numRivers
FROM country C, mountainInfo MI, riverInfo RI
WHERE C.code = MI.country AND C.code = RI.country
ORDER BY numMountains + numRivers DESC;

--9
WITH lakeProvince AS (SELECT GL.province, MIN(L.elevation) as minElevation
FROM geo_lake GL, lake L
WHERE GL.lake = L.name AND L.elevation IS NOT NULL
GROUP BY GL.province),

mountainProvince AS (SELECT GM.province, MAX(M.elevation) as maxElevation
FROM geo_mountain GM, mountain M
WHERE GM.mountain = M.name AND M.elevation IS NOT NULL
GROUP BY GM.province)

SELECT L.province, C.name, M.maxElevation, L.minElevation, M.maxElevation - L.minElevation AS elevDiff
FROM lakeProvince L, province P, country C, mountainProvince M
WHERE L.province = P.name
    AND P.country = C.code
    AND M.province = L.province
    AND M.maxElevation - L.minElevation > 4500
ORDER BY elevDiff DESC;

--10
WITH fastGrowingCountry AS (SELECT country, population_growth
    FROM population
    WHERE population_growth > 2),

countryOrgs AS (SELECT M.country, COUNT(organization) AS orgCount
    FROM isMember M, fastGrowingCountry C
    WHERE M.country = C.country
    GROUP BY M.country)
   
SELECT C.country, O.orgCount
FROM fastGrowingCountry C, countryOrgs O
WHERE O.orgCount = (SELECT MAX(orgCount) FROM countryOrgs)
    AND C.country = O.country;
    
--11
WITH riverInfo AS (SELECT E.continent, RI.island, RI.river, R.length
    FROM RiverOnIsland RI, river R, geo_river GR, encompasses E
    WHERE RI.river = R.name
        AND R.name = GR.river
        AND GR.country = E.country),
        
longestRiver AS (SELECT island, MAX(length) as longest
    FROM riverInfo
    GROUP BY island)
    
SELECT continent, R.island, river, length
FROM riverInfo R, longestRiver L
WHERE R.island = L.island
    AND R.length = L.longest
ORDER BY continent, length DESC;

--12
WITH largePops AS (SELECT code
    FROM country
    WHERE population > 500000000),

largeDeserts AS (SELECT province, MAX(area) AS maxArea
    FROM desert D, geo_desert GD
    WHERE D.name = GD.desert
    GROUP BY province)
    
SELECT DISTINCT LD.province, C.name as Country, GD.desert
FROM largeDeserts LD, province P, desert D, geo_desert GD, largePops LP, country C
WHERE LD.province = P.name
    AND LD.maxArea = D.area
    AND GD.province = LD.province
    AND GD.desert = D.name
    AND P.country = LP.code
    AND P.country = C.code
ORDER BY province, country;

--13
WITH qualCountry AS (SELECT code
    FROM country
    WHERE population > 50000000),
    
eGroups AS (SELECT country, MAX(percentage) AS maxPercentage
    FROM ethnicGroup
    WHERE percentage > 55
    GROUP BY country)
    
SELECT DISTINCT C.name, G.name, G.percentage
FROM qualCountry Q, eGroups E, ethnicGroup G, country C
WHERE Q.code = E.country
    AND E.maxPercentage = G.percentage
    AND G.country = Q.code
    AND Q.code = C.code
ORDER BY C.name, G.name;
--Note: because the min requirement for group inclusion was 55% of the population
-- I ignored the requirement for possibly listing multiple groups per country
-- As it was not possible to have more than 1 such qualified group

--14
WITH qualLang AS (SELECT country, percentage
    FROM language
    WHERE percentage >= 5),
    
langAgg AS (SELECT country, COUNT(percentage) AS langCount, MAX(percentage) as maxPerc
    FROM qualLang
    GROUP BY country
    HAVING COUNT(percentage) >= 4)
    
SELECT C.name, LA.langCount, L.name, L.percentage
FROM langAgg LA, country C, language L
WHERE LA.country = C.code
    AND L.country = LA.country
    AND LA.maxPerc = L.percentage
ORDER BY langCount DESC, C.name;

--15
SELECT C.name
FROM religion R1, religion R2, country C
WHERE R1.country = R2.country
    AND R1.name = 'Muslim'
    AND R2.name = 'Christian'
    AND R2.percentage > R1.percentage
    AND R1.country = C.code
ORDER BY C.name;

--16
WITH uCountries AS (SELECT M.country
FROM organization O, isMember M
WHERE O.name = 'European Union'
    AND O.abbreviation = M.organization)
    
SELECT P.name, P.population, C.name
FROM uCountries U, province P, country C
WHERE U.country = P.country
    AND P.population > 10000000        
    AND U.country = C.code;

--17
WITH capitalPop AS (SELECT C.code, C2.name, C2.population
    FROM Country C, city C2
    WHERE C.capital = C2.name
        AND C.capital IS NOT NULL
        AND C2.population IS NOT NULL
        AND C.code = C2.country),
    
numCities AS (SELECT country, COUNT(name)
    FROM City
    GROUP BY country
    HAVING COUNT(name) > 1),
    
otherCities AS (SELECT C.country, name, population
        FROM City C, numCities N
        WHERE C.country = N.country
            AND population IS NOT NULL
        MINUS
        SELECT *
        FROM capitalPop),
        
avgPop AS (SELECT country, AVG(population) AS averagePop
    FROM otherCities
    GROUP BY country)
    

SELECT C2.name, ROUND(A.averagePop, 2) AS avgPopulation, C.population AS capitalPop
FROM capitalPop C, avgPop A, country C2
WHERE C.code = A.country
    AND C.code = C2.code
ORDER BY capitalPop - avgPopulation DESC;

--18
WITH countryHighest AS (SELECT country, MAX(elevation) as highestElev
    FROM geo_mountain GM, mountain M
    WHERE GM.mountain = M.name
    GROUP BY country),
    
highMountContinent AS (SELECT E.continent, MAX(elevation) as highestElev
    FROM geo_mountain GM, mountain M, encompasses E
    WHERE GM.mountain = M.name
        AND GM.country = E.country
    GROUP BY E.continent),
    
highestCountry AS (SELECT CH.country, CH.highestElev
    FROM countryHighest CH, highMountContinent HM, encompasses E
    WHERE E.country = CH.country
        AND E.continent = HM.continent
        AND CH.highestElev = HM.highestElev),
        
countryLongest AS (SELECT country, MAX(length) as longest
    FROM geo_river GR, river R
    WHERE GR.river = R.name
    GROUP BY country),
    
longestContinent AS (SELECT E.continent, MAX(length) as longest
    FROM geo_river GR, river R, encompasses E
    WHERE GR.river = R.name
        AND GR.country = E.country
    GROUP BY E.continent),
    
longestCountry AS (SELECT CL.country, CL.longest
    FROM countryLongest CL, longestContinent LC, encompasses E
    WHERE E.country = CL.country
        AND E.continent = LC.continent
        AND CL.longest = LC.longest)
    
SELECT *
FROM longestCountry LC, highestCountry HC
WHERE LC.country = HC.country;