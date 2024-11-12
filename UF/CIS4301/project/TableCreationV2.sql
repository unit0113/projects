CREATE TABLE Releases
(
	ReleaseID   		INT NOT NULL,
	Year				INT,
	Facility			VARCHAR2(200) NOT NULL,
	Industry			VARCHAR2(100) NOT NULL,
    ChemicalID          INT NOT NULL,
    TRIFID              VARCHAR2(50) NOT NULL,
    Unit                VARCHAR2(25) NOT NULL,
    ReleaseTotal        INT NOT NULL,
	PRIMARY KEY(ReleaseID)
);

CREATE TABLE Chemicals
(
	ChemicalID			INT NOT NULL,
	ChemicalType		VARCHAR2(1000) NOT NULL,
	HealthEffects		VARCHAR2(1000),
	PRIMARY KEY(ChemicalID)
);

CREATE TABLE Sites
(
	TRIFID              VARCHAR2(50) NOT NULL,
    FRSID               INT,
	Longitude			DOUBLE PRECISION,
	Latitude			DOUBLE PRECISION,
	Street				VARCHAR2(100),
	City				VARCHAR2(50),
    State               VARCHAR2(2) NOT NULL,
	County			    VARCHAR2(50) NOT NULL,
	ZipCode			    VARCHAR2(9) NOT NULL,
	PRIMARY KEY(TRIFID)
);

CREATE TABLE StateGov
(
	Abbreviation        VARCHAR2(2) NOT NULL,
    Year                INT NOT NULL,
	GoverningParty		CHAR NOT NULL,
	PRIMARY KEY(Abbreviation, Year)
);

CREATE TABLE FIPS
(
	Abbreviation        VARCHAR2(2) NOT NULL,
    FIPSCode			INT NOT NULL,
	PRIMARY KEY(Abbreviation)
);

CREATE TABLE Legislatures
(
	Year       			INT NOT NULL,
	House	  		    CHAR NOT NULL,
    Seneate	  		    CHAR NOT NULL,
	PRIMARY KEY(Year)
);

CREATE TABLE Presidents
(
	Year       			INT NOT NULL,
    Name				VARCHAR2(50) NOT NULL,
	Party				CHAR NOT NULL,
	PRIMARY KEY(Year)
);
