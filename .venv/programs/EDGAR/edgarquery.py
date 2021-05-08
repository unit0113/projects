import requests
from bs4 import BeautifulSoup

class StockData:

    """ Class to store stock data for requested tickers, takes in initial input from get_tickers
    """    
    
    def __init__(self, symbol, cik):
        """ Takes in ticker from get_tickers output and initiates the class

        Args:
            symbol (str): Stocker ticker
        """ 

        self.symbol = symbol
        self.cik = cik

def filingsList(cik):
    """Creates URL based on CIK number, and pulls the complete list of EDGAR filings

    Args:
        cik (int): CIK number for the stock

    Returns:
        filings (dict): complete list of company fillings
    """
    # Assemble URL
    url = "https://www.sec.gov/Archives/edgar/data/" + str(cik) + "/index.json"

    # Get the filings list
    filings = requests.get(url).json()
    return filings

def getfilings(filings, cik):

    # Loop through list
    for filing in filings['directory']['item']:

        # Get URL for individual filing
        filingNumber = filing['name']
        filingURL = "https://www.sec.gov/Archives/edgar/data/" + str(cik) + "/" + str(filingNumber) + "/index.json"
        filingData = requests.get(filingURL).json()
        
        # Pull data from filing
        for doc in filingData['directory']['item']:
            if doc['type'] != 'image2.gif':
                docName = doc['name']
                docURL = "https://www.sec.gov/Archives/edgar/data/" + str(cik) + "/" + str(filingNumber) + "/" + docName
                print(docURL)


def quarterlyData(cik):
    pass

def annualData(cik):
    pass

def main(guiReturn):
    """ Runs two subprograms. Webscrapes ticker/CIK paires from SEC site, gets tickers from user and passes the CIK codes and a seperate flag to the edgarquery program

    Returns:
        cik1 (str): first CIK input
        cik2 (str): second CIK input
        excelFlag (bool): excel export flag
    """
    # Pull data from GUI and initiate StockData class(es)
    stock2Flag = False
    if guiReturn[0] == "":
        stock1 = StockData(guiReturn[2], guiReturn[3])
    else:
        stock1 = StockData(guiReturn[0], guiReturn[1])
        if guiReturn[2] != "":
            stock2 = StockData(guiReturn[2], guiReturn[3])
            stock2Flag = True            
    excelFlag = guiReturn[4]

    # Get list of fillings
    filings1 = filingsList(stock1.cik)
    if stock2Flag:
        filings2 = filingsList(stock2.cik)

    # Pull fillings
    getfilings(filings1, stock1.cik)
    


    
#guiReturn = []
if __name__ == "__main__":
    main(guiReturn)