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

def fillingsList(cik):
    """Creates URL based on CIK number, and pulls the complete list of EDGAR filings

    Args:
        cik (int): CIK number for the stock

    Returns:
        fillings (dict): complete list of company fillings
    """
    # Assemble URL
    url = "https://www.sec.gov/Archives/edgar/data/" + str(cik) + "/index.json"

    # Get the fillings list
    fillings = requests.get(url).json()
    return fillings

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
    fillings1 = fillingsList(stock1.cik)
    if stock2Flag:
        fillings2 = fillingsList(stock2.cik)
    


    
#guiReturn = []
if __name__ == "__main__":
    main(guiReturn)