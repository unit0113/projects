ALTER SESSION SET NLS_DATE_FORMAT = 'DD MM SYYYY';
-- SYYYY means 4-digit-year, S prefixes BC years with "-"

CREATE TABLE Country
(Name VARCHAR2(50) NOT NULL UNIQUE,
 Code VARCHAR2(4) CONSTRAINT CountryKey PRIMARY KEY,
 Capital VARCHAR2(50),
 Province VARCHAR2(50),
 Area NUMBER CONSTRAINT CountryArea
   CHECK (Area >= 0),
 Population NUMBER CONSTRAINT CountryPop
   CHECK (Population >= 0));

CREATE TABLE City
(Name VARCHAR2(50),
 Country VARCHAR2(4),
 Province VARCHAR2(50),
 Population NUMBER CONSTRAINT CityPop
   CHECK (Population >= 0),
 Latitude NUMBER CONSTRAINT CityLat
   CHECK ((Latitude >= -90) AND (Latitude <= 90)),
 Longitude NUMBER CONSTRAINT CityLon
   CHECK ((Longitude >= -180) AND (Longitude <= 180)),
 Elevation NUMBER,
 CONSTRAINT CityKey PRIMARY KEY (Name, Country, Province));

CREATE TABLE Province
(Name VARCHAR2(50) CONSTRAINT PrName NOT NULL,
 Country  VARCHAR2(4) CONSTRAINT PrCountry NOT NULL,
 Population NUMBER CONSTRAINT PrPop
   CHECK (Population >= 0),
 Area NUMBER CONSTRAINT PrAr
   CHECK (Area >= 0),
 Capital VARCHAR2(50),
 CapProv VARCHAR2(50),
 CONSTRAINT PrKey PRIMARY KEY (Name, Country));

ALTER TABLE Country ADD CONSTRAINT FK_CountryREFCity
  FOREIGN KEY (Code, Capital, Province) REFERENCES City(Country, Name, Province)
  INITIALLY DEFERRED DEFERRABLE;

ALTER TABLE City ADD CONSTRAINT FK_CityREFProvince
  FOREIGN KEY (Country, Province) REFERENCES Province(Country, Name)
  INITIALLY DEFERRED DEFERRABLE;

ALTER TABLE Province ADD CONSTRAINT FK_ProvinceREFCountry
  FOREIGN KEY (Country) REFERENCES Country(Code)
  INITIALLY DEFERRED DEFERRABLE;

ALTER TABLE Province ADD CONSTRAINT FK_ProvinceREFCity
  FOREIGN KEY (Capital, Country, CapProv) REFERENCES City(Name, Country, Province)
  INITIALLY DEFERRED DEFERRABLE;

CREATE TABLE Economy
(Country VARCHAR2(4) CONSTRAINT EconomyKey PRIMARY KEY,
 GDP NUMBER CONSTRAINT EconomyGDP
   CHECK (GDP >= 0),
 Agriculture NUMBER,
 Service NUMBER,
 Industry NUMBER,
 Inflation NUMBER,
 Unemployment NUMBER,
 CONSTRAINT FK_EconomyREFCountry FOREIGN KEY (Country) REFERENCES Country(Code));

CREATE TABLE Population
(Country VARCHAR2(4) CONSTRAINT PopKey PRIMARY KEY,
 Population_Growth NUMBER,
 Infant_Mortality NUMBER,
 CONSTRAINT FK_CountryPop FOREIGN KEY (Country) REFERENCES Country(code));

CREATE TABLE Politics
(Country VARCHAR2(4) CONSTRAINT PoliticsKey PRIMARY KEY,
 Independence DATE,
 WasDependent VARCHAR2(50),
 Dependent  VARCHAR2(4),
 Government VARCHAR2(120),
 CONSTRAINT FK_PoliticsCountryREFCountry FOREIGN KEY (Country) REFERENCES Country(Code),
 CONSTRAINT FK_PoliticsDependentREFCountry FOREIGN KEY (Dependent) REFERENCES Country(Code));

CREATE TABLE Language
(Country VARCHAR2(4),
 Name VARCHAR2(50),
 Percentage NUMBER CONSTRAINT LanguagePercent 
   CHECK ((Percentage > 0) AND (Percentage <= 100)),
 CONSTRAINT LanguageKey PRIMARY KEY (Name, Country),
 CONSTRAINT FK_LanguageREFCountry FOREIGN KEY (Country) REFERENCES Country(Code));

CREATE TABLE Religion
(Country VARCHAR2(4),
 Name VARCHAR2(50),
 Percentage NUMBER CONSTRAINT ReligionPercent 
   CHECK ((Percentage > 0) AND (Percentage <= 100)),
 CONSTRAINT ReligionKey PRIMARY KEY (Name, Country),
 CONSTRAINT FK_ReligionREFCountry FOREIGN KEY (Country) REFERENCES Country(Code));

CREATE TABLE EthnicGroup
(Country VARCHAR2(4),
 Name VARCHAR2(50),
 Percentage NUMBER CONSTRAINT EthnicPercent 
   CHECK ((Percentage > 0) AND (Percentage <= 100)),
 CONSTRAINT EthnicKey PRIMARY KEY (Name, Country),
 CONSTRAINT FK_EthnicGroupREFCountry FOREIGN KEY (Country) REFERENCES Country(Code));

CREATE TABLE Countrypops
(Country VARCHAR2(4),
 Year NUMBER CONSTRAINT CountryPopsYear
   CHECK (Year >= 0),
 Population NUMBER CONSTRAINT CountryPopsPop
   CHECK (Population >= 0),
 CONSTRAINT CountryPopsKey PRIMARY KEY (Country, Year),
 CONSTRAINT FK_CountrypopsREFCountry FOREIGN KEY (Country) REFERENCES Country(Code));

CREATE TABLE Countryothername
(Country VARCHAR2(4),
 othername VARCHAR2(50),
 CONSTRAINT CountryOthernameKey PRIMARY KEY (Country, othername),
 CONSTRAINT FK_CountryothernameREFCountry FOREIGN KEY (Country) REFERENCES Country(Code));

CREATE TABLE Countrylocalname
(Country VARCHAR2(4),
 localname VARCHAR2(120),
 CONSTRAINT CountrylocalnameKey PRIMARY KEY (Country),
 CONSTRAINT FK_CountrylocalnameREFCountry FOREIGN KEY (Country) REFERENCES Country(Code));

CREATE TABLE Provpops
(Province VARCHAR2(50),
 Country VARCHAR2(4),
 Year NUMBER CONSTRAINT ProvPopsYear
   CHECK (Year >= 0),
 Population NUMBER CONSTRAINT ProvPopsPop
   CHECK (Population >= 0),
 CONSTRAINT ProvPopKey PRIMARY KEY (Country, Province, Year),
 CONSTRAINT FK_ProvpopsREFProvince FOREIGN KEY (Province, Country) REFERENCES Province(Name, Country));

CREATE TABLE Provinceothername
(Province VARCHAR2(50),
 Country VARCHAR2(4),
 othername VARCHAR2(50),
 CONSTRAINT ProvOthernameKey PRIMARY KEY (Country, Province, othername),
 CONSTRAINT FK_ProvinceothernameREFProvince FOREIGN KEY (Province, Country) REFERENCES Province(Name, Country));

CREATE TABLE Provincelocalname
(Province VARCHAR2(50),
 Country VARCHAR2(4),
 localname VARCHAR2(120),
 CONSTRAINT ProvlocalnameKey PRIMARY KEY (Country, Province),
 CONSTRAINT FK_ProvincelocalnameREFProvince FOREIGN KEY (Province, Country) REFERENCES Province(Name, Country));

CREATE TABLE Citypops
(City VARCHAR2(50),
 Country VARCHAR2(4),
 Province VARCHAR2(50),
 Year NUMBER CONSTRAINT CityPopsYear
   CHECK (Year >= 0),
 Population NUMBER CONSTRAINT CityPopsPop
   CHECK (Population >= 0),
 CONSTRAINT CityPopKey PRIMARY KEY (Country, Province, City, Year),
 CONSTRAINT FK_CitypopsREFCity FOREIGN KEY (City, Country, Province) REFERENCES City(Name, Country, Province));

CREATE TABLE Cityothername
(City VARCHAR2(50),
 Country VARCHAR2(4),
 Province VARCHAR2(50),
 othername VARCHAR2(50),
 CONSTRAINT CityOthernameKey PRIMARY KEY (Country, Province, City, othername),
 CONSTRAINT FK_CityothernameREFCity FOREIGN KEY (City, Country, Province) REFERENCES City(Name, Country, Province));

CREATE TABLE Citylocalname
(City VARCHAR2(50),
 Country VARCHAR2(4),
 Province VARCHAR2(50),
 localname VARCHAR2(120),
 CONSTRAINT CitylocalnameKey PRIMARY KEY (Country, Province, City),
 CONSTRAINT FK_CitylocalnameREFCity FOREIGN KEY (City, Country, Province) REFERENCES City(Name, Country, Province));

CREATE TABLE Continent
(Name VARCHAR2(20) CONSTRAINT ContinentKey PRIMARY KEY,
 Area NUMBER(10));

CREATE TABLE borders
(Country1 VARCHAR2(4),
 Country2 VARCHAR2(4),
 Length NUMBER 
   CHECK (Length > 0),
 CONSTRAINT BorderKey PRIMARY KEY (Country1, Country2),
 CONSTRAINT bordersCountries CHECK (Country1 != Country2),
 CONSTRAINT FK_Country1REFCountry FOREIGN KEY (Country1) REFERENCES Country(Code),
 CONSTRAINT FK_Country2REFCountry FOREIGN KEY (Country2) REFERENCES Country(Code));

CREATE TABLE encompasses
(Country VARCHAR2(4) NOT NULL,
 Continent VARCHAR2(20) NOT NULL,
 Percentage NUMBER,
   CHECK ((Percentage > 0) AND (Percentage <= 100)),
 CONSTRAINT EncompassesKey PRIMARY KEY (Country,Continent),
 CONSTRAINT FK_encompassesREFCountry FOREIGN KEY (Country) REFERENCES Country(Code),
 CONSTRAINT FK_encompassesREFContinent FOREIGN KEY (Continent) REFERENCES Continent(Name));

CREATE TABLE Organization
(Abbreviation VARCHAR2(12) Constraint OrgKey PRIMARY KEY,
 Name VARCHAR2(100) NOT NULL,
 City VARCHAR2(50),
 Country VARCHAR2(4), 
 Province VARCHAR2(50),
 Established DATE,
 CONSTRAINT OrgNameUnique UNIQUE (Name),
 CONSTRAINT FK_OrganizationREFCity FOREIGN KEY (City, Country, Province) REFERENCES City(Name, Country, Province));

CREATE TABLE isMember
(Country VARCHAR2(4),
 Organization VARCHAR2(12),
 Type VARCHAR2(60) DEFAULT 'member',
 CONSTRAINT MemberKey PRIMARY KEY (Country,Organization),
 CONSTRAINT FK_isMemberREFCountry FOREIGN KEY (Country) REFERENCES Country(Code),
 CONSTRAINT FK_isMemberREFOrganization FOREIGN KEY (Organization) REFERENCES Organization(Abbreviation));

CREATE OR REPLACE TYPE GeoCoord AS OBJECT
(Latitude NUMBER,
 Longitude NUMBER);
/

CREATE TABLE Mountain
(Name VARCHAR2(50) CONSTRAINT MountainKey PRIMARY KEY,
 Mountains VARCHAR2(50),
 Elevation NUMBER,
 Type VARCHAR2(10),
 Coordinates GeoCoord CONSTRAINT MountainCoord
     CHECK ((Coordinates.Latitude >= -90) AND 
            (Coordinates.Latitude <= 90) AND
            (Coordinates.Longitude > -180) AND
            (Coordinates.Longitude <= 180)));

CREATE TABLE Desert
(Name VARCHAR2(50) CONSTRAINT DesertKey PRIMARY KEY,
 Area NUMBER,
 Coordinates GeoCoord CONSTRAINT DesCoord
     CHECK ((Coordinates.Latitude >= -90) AND 
            (Coordinates.Latitude <= 90) AND
            (Coordinates.Longitude > -180) AND
            (Coordinates.Longitude <= 180)));

CREATE TABLE Island
(Name VARCHAR2(50) CONSTRAINT IslandKey PRIMARY KEY,
 Islands VARCHAR2(50),
 Area NUMBER CONSTRAINT IslandAr check (Area >= 0),
 Elevation NUMBER,
 Type VARCHAR2(10),
 Coordinates GeoCoord CONSTRAINT IslandCoord
     CHECK ((Coordinates.Latitude >= -90) AND 
            (Coordinates.Latitude <= 90) AND
            (Coordinates.Longitude > -180) AND
            (Coordinates.Longitude <= 180)));

CREATE TABLE Sea
(Name VARCHAR2(50) CONSTRAINT SeaKey PRIMARY KEY,
 Area NUMBER CONSTRAINT SeaAr CHECK (Area >= 0),
 Depth NUMBER CONSTRAINT SeaDepth CHECK (Depth >= 0));

CREATE TABLE Lake
(Name VARCHAR2(50) CONSTRAINT LakeKey PRIMARY KEY,
 River VARCHAR2(50),
 Area NUMBER CONSTRAINT LakeAr CHECK (Area >= 0),
 Elevation NUMBER,
 Depth NUMBER CONSTRAINT LakeDpth CHECK (Depth >= 0),
 Height NUMBER CONSTRAINT DamHeight CHECK (Height > 0),
 Type VARCHAR2(12),
 Coordinates GeoCoord CONSTRAINT LakeCoord
     CHECK ((Coordinates.Latitude >= -90) AND 
            (Coordinates.Latitude <= 90) AND
            (Coordinates.Longitude > -180) AND
            (Coordinates.Longitude <= 180)));

CREATE TABLE River
(Name VARCHAR2(50) CONSTRAINT RiverKey PRIMARY KEY,
 River VARCHAR2(50),
 Lake VARCHAR2(50),
 Sea VARCHAR2(50),
 Length NUMBER CONSTRAINT RiverLength
   CHECK (Length >= 0),
 Area NUMBER CONSTRAINT RiverArea
   CHECK (Area >= 0),
 Source GeoCoord CONSTRAINT SourceCoord
     CHECK ((Source.Latitude >= -90) AND 
            (Source.Latitude <= 90) AND
            (Source.Longitude > -180) AND
            (Source.Longitude <= 180)),
 Mountains VARCHAR2(50),
 SourceElevation NUMBER,
 Estuary GeoCoord CONSTRAINT EstCoord
     CHECK ((Estuary.Latitude >= -90) AND 
            (Estuary.Latitude <= 90) AND
            (Estuary.Longitude > -180) AND
            (Estuary.Longitude <= 180)),
 EstuaryElevation NUMBER,
 CONSTRAINT RivFlowsInto 
     CHECK ((River IS NULL AND Lake IS NULL)
            OR (River IS NULL AND Sea IS NULL)
            OR (Lake IS NULL AND Sea is NULL)),
 CONSTRAINT RiverNameRiverRiver CHECK (Name != River),
 CONSTRAINT FK_RiverRiverREFRiverName FOREIGN KEY (River) REFERENCES River(Name),
 CONSTRAINT FK_RiverREFSea FOREIGN KEY (Sea) REFERENCES Sea(Name));

ALTER TABLE Lake ADD CONSTRAINT FK_LakeREFRiver
  FOREIGN KEY (River) REFERENCES River(Name)
  INITIALLY DEFERRED DEFERRABLE;

ALTER TABLE River ADD CONSTRAINT FK_RiverREFLake
  FOREIGN KEY (Lake) REFERENCES Lake(Name)
  INITIALLY DEFERRED DEFERRABLE;

CREATE TABLE RiverThrough
(River VARCHAR2(50),
 Lake  VARCHAR2(50),
 CONSTRAINT RThroughKey PRIMARY KEY (River,Lake),
 CONSTRAINT FK_RiverThroughREFRiver FOREIGN KEY (River) REFERENCES River(Name),
 CONSTRAINT FK_RiverThroughREFLake FOREIGN KEY (Lake) REFERENCES Lake(Name));

CREATE TABLE geo_Mountain
(Mountain VARCHAR2(50),
 Country VARCHAR2(4),
 Province VARCHAR2(50),
 CONSTRAINT GMountainKey PRIMARY KEY (Province,Country,Mountain),
 CONSTRAINT FK_geo_MountainREFMountain FOREIGN KEY (Mountain) REFERENCES Mountain(Name),
 CONSTRAINT FK_geo_MountainREFProvince FOREIGN KEY (Province, Country) REFERENCES Province(Name, Country));

CREATE TABLE geo_Desert
(Desert VARCHAR2(50),
 Country VARCHAR2(4),
 Province VARCHAR2(50),
 CONSTRAINT GDesertKey PRIMARY KEY (Province, Country, Desert),
 CONSTRAINT FK_geo_DesertREFDesert FOREIGN KEY (Desert) REFERENCES Desert(Name),
 CONSTRAINT FK_geo_DesertREFProvince FOREIGN KEY (Province, Country) REFERENCES Province(Name, Country));

CREATE TABLE geo_Island
(Island VARCHAR2(50), 
 Country VARCHAR2(4),
 Province VARCHAR2(50),
 CONSTRAINT GIslandKey PRIMARY KEY (Province, Country, Island),
 CONSTRAINT FK_geo_IslandREFIsland FOREIGN KEY (Island) REFERENCES Island(Name),
 CONSTRAINT FK_geo_IslandREFProvince FOREIGN KEY (Province, Country) REFERENCES Province(Name, Country));

CREATE TABLE geo_River
(River VARCHAR2(50), 
 Country VARCHAR2(4),
 Province VARCHAR2(50),
 CONSTRAINT GRiverKey PRIMARY KEY (Province, Country, River),
 CONSTRAINT FK_geo_RiverREFRiver FOREIGN KEY (River) REFERENCES River(Name),
 CONSTRAINT FK_geo_RiverREFProvince FOREIGN KEY (Province, Country) REFERENCES Province(Name, Country));

CREATE TABLE geo_Sea
(Sea VARCHAR2(50),
 Country VARCHAR2(4),
 Province VARCHAR2(50),
 CONSTRAINT GSeaKey PRIMARY KEY (Province, Country, Sea),
 CONSTRAINT FK_geo_SeaREFSea FOREIGN KEY (Sea) REFERENCES Sea(Name),
 CONSTRAINT FK_geo_SeaREFProvince FOREIGN KEY (Province, Country) REFERENCES Province(Name, Country));

CREATE TABLE geo_Lake
(Lake VARCHAR2(50),
 Country VARCHAR2(4),
 Province VARCHAR2(50),
 CONSTRAINT GLakeKey PRIMARY KEY (Province, Country, Lake),
 CONSTRAINT FK_geo_LakeREFLake FOREIGN KEY (Lake) REFERENCES Lake(Name),
 CONSTRAINT FK_geo_LakeREFProvince FOREIGN KEY (Province, Country) REFERENCES Province(Name, Country));

CREATE TABLE geo_Source
(River VARCHAR2(50),
 Country VARCHAR2(4),
 Province VARCHAR2(50),
 CONSTRAINT GSourceKey PRIMARY KEY (Province, Country, River),
 CONSTRAINT FK_geo_SourceREFRiver FOREIGN KEY (River) REFERENCES River(Name),
 CONSTRAINT FK_geo_SourceREFProvince FOREIGN KEY (Province, Country) REFERENCES Province(Name, Country));

CREATE TABLE geo_Estuary
(River VARCHAR2(50),
 Country VARCHAR2(4),
 Province VARCHAR2(50),
 CONSTRAINT GEstuaryKey PRIMARY KEY (Province, Country, River),
 CONSTRAINT FK_geo_EstuaryREFRiver FOREIGN KEY (River) REFERENCES River(Name),
 CONSTRAINT FK_geo_EstuaryREFProvince FOREIGN KEY (Province, Country) REFERENCES Province(Name, Country));

CREATE TABLE mergesWith
(Sea1 VARCHAR2(50),
 Sea2 VARCHAR2(50),
 CONSTRAINT MergesWithKey PRIMARY KEY (Sea1, Sea2),
 CONSTRAINT sea_chk CHECK (Sea1 != Sea2),
 CONSTRAINT FK_mergesWithSea1REFSea FOREIGN KEY (Sea1) REFERENCES Sea(Name),
 CONSTRAINT FK_mergesWithSea2REFSea FOREIGN KEY (Sea2) REFERENCES Sea(Name));

CREATE TABLE located
(City VARCHAR2(50),
 Province VARCHAR2(50),
 Country VARCHAR2(4),
 River VARCHAR2(50),
 Lake VARCHAR2(50),
 Sea VARCHAR2(50),
 CONSTRAINT FK_locatedREFCity FOREIGN KEY (City, Province, Country) REFERENCES City(Name, Province, Country),
 CONSTRAINT FK_locatedREFRiver FOREIGN KEY (River) REFERENCES River(Name),
 CONSTRAINT FK_locatedREFLake FOREIGN KEY (Lake) REFERENCES Lake(Name),
 CONSTRAINT FK_locatedREFSea FOREIGN KEY (Sea) REFERENCES Sea(Name));

CREATE TABLE locatedOn
(City VARCHAR2(50),
 Province VARCHAR2(50),
 Country VARCHAR2(4),
 Island VARCHAR2(50),
 CONSTRAINT locatedOnKey PRIMARY KEY (City, Province, Country, Island),
 CONSTRAINT FK_locatedOnREFCity FOREIGN KEY (City, Province, Country) REFERENCES City(Name, Province, Country),
 CONSTRAINT FK_locatedREFIsland FOREIGN KEY (Island) REFERENCES Island(Name));

CREATE TABLE islandIn
(Island VARCHAR2(50),
 Sea VARCHAR2(50),
 Lake VARCHAR2(50),
 River VARCHAR2(50),
 CONSTRAINT FK_islandInREFIsland FOREIGN KEY (Island) REFERENCES Island(Name),
 CONSTRAINT FK_islandInREFSea FOREIGN KEY (Sea) REFERENCES Sea(Name),
 CONSTRAINT FK_islandInREFLake FOREIGN KEY (Lake) REFERENCES Lake(Name),
 CONSTRAINT FK_islandInREFRiver FOREIGN KEY (River) REFERENCES River(Name));

CREATE TABLE MountainOnIsland
(Mountain VARCHAR2(50),
 Island   VARCHAR2(50),
 CONSTRAINT MountainIslKey PRIMARY KEY (Mountain, Island),
 CONSTRAINT FK_MountainOnIslandREFMountain FOREIGN KEY (Mountain) REFERENCES Mountain(Name),
 CONSTRAINT FK_MountainOnIslandREFIsland FOREIGN KEY (Island) REFERENCES Island(Name));

CREATE TABLE LakeOnIsland
(Lake    VARCHAR2(50),
 Island  VARCHAR2(50),
 CONSTRAINT LakeIslKey PRIMARY KEY (Lake, Island),
 CONSTRAINT FK_LakeOnIslandREFLake FOREIGN KEY (Lake) REFERENCES Lake(Name),
 CONSTRAINT FK_LakeOnIslandREFIsland FOREIGN KEY (Island) REFERENCES Island(Name));

CREATE TABLE RiverOnIsland
(River   VARCHAR2(50),
 Island  VARCHAR2(50),
 CONSTRAINT RiverIslKey PRIMARY KEY (River, Island),
 CONSTRAINT FK_RiverOnIslandREFRiver FOREIGN KEY (River) REFERENCES River(Name),
 CONSTRAINT FK_RiverOnIslandREFIsland FOREIGN KEY (Island) REFERENCES Island(Name));

CREATE TABLE Airport
(IATACode VARCHAR(3) PRIMARY KEY,
 Name VARCHAR2(100),
 Country VARCHAR2(4),
 City VARCHAR2(50),
 Province VARCHAR2(50),
 Island VARCHAR2(50),
 Latitude NUMBER CONSTRAINT AirpLat
   CHECK ((Latitude >= -90) AND (Latitude <= 90)),
 Longitude NUMBER CONSTRAINT AirpLon
   CHECK ((Longitude >= -180) AND (Longitude <= 180)),
 Elevation NUMBER,
 gmtOffset NUMBER,
 CONSTRAINT FK_AirportREFCountry FOREIGN KEY (Country) REFERENCES Country(Code),
 CONSTRAINT FK_AirportREFIsland FOREIGN KEY (Island) REFERENCES Island(Name),
 CONSTRAINT FK_AirportREFCity FOREIGN KEY (City, Country, Province) REFERENCES City(Name, Country, Province));
