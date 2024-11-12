CREATE TABLE Releases
(
	TRIID				INT NOT NULL,
	Year				INT,
	OffendingCompany	VARCHAR2(50) NOT NULL,
	Industry			VARCHAR2(20) NOT NULL,
	PRIMARY KEY(TRIID)
);

CREATE TABLE Chemicals
(
	ChemicalID			INT NOT NULL,
	ChemicalType		VARCHAR2(20) NOT NULL,
	HealthEffects		VARCHAR2(20),
	PRIMARY KEY(ChemicalID)
);

CREATE TABLE Addresses
(
	AddressID           INT NOT NULL,
    AddressNumber	    INT NOT NULL,
	Street				VARCHAR2(20) NOT NULL,
	City				VARCHAR2(20) NOT NULL,
    State               VARCHAR2(2) NOT NULL,
	County			    VARCHAR2(20) NOT NULL,
	ZipCode			    VARCHAR2(20) NOT NULL,
	PRIMARY KEY(AddressID)
);

CREATE TABLE Sites
(
	SiteID				INT NOT NULL,
	Longitude			DOUBLE PRECISION,
	Latitude			DOUBLE PRECISION,
    AddressID           INT NOT NULL,
	PRIMARY KEY(SiteID)
);

CREATE TABLE States
(
	Abbreviation        VARCHAR2(2) NOT NULL,
    FIPSCode			INT NOT NULL,
	GoverningParty		VARCHAR2(20) NOT NULL,
	PRIMARY KEY(Abbreviation)
);

CREATE TABLE Legislatures
(
	LegislatureID			INT NOT NULL,
	MajorityParty			VARCHAR2(20) NOT NULL,
	PRIMARY KEY(LegislatureID)
);

CREATE TABLE Presidents
(
	Name				VARCHAR2(20) NOT NULL,
	Party				VARCHAR2(20) NOT NULL,
	PRIMARY KEY(Name)
);
