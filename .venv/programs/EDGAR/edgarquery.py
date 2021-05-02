

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





def main(guiReturn):
    """ Runs two subprograms. Webscrapes ticker/CIK paires from SEC site, gets tickers from user and passes the CIK codes and a seperate flag to the edgarquery program

    Returns:
        cik1 (str): first CIK input
        cik2 (str): second CIK input
        excelFlag (bool): excel export flag
    """
    print(guiReturn)


if __name__ == "__main__":
    main()