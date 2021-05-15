import requests
from datetime import date
from bs4 import BeautifulSoup
import pandas as pd

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

def getFilings(cik, period):
    # Base URL for SEC EDGAR browser
    endpoint = r"https://www.sec.gov/cgi-bin/browse-edgar"

    # Define parameters dictionary
    param_dict = {'action':'getcompany',
                'CIK':cik,
                'type':period,
                'dateb':'',
                'owner':'exclude',
                'start':'',
                'output':'',
                'count':'100'}

    # Get data
    response = requests.get(url = endpoint, params = param_dict)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Build document table
    doc_table = soup.find_all('table', class_='tableFile2')

    # Base URL for link building
    base_url_sec = r"https://www.sec.gov"

    # Loop through rows in table
    master_list = []
    today = date.today()
    curYear = today.year

    for row in doc_table[0].find_all('tr'):
        
        # Find columns
        cols = row.find_all('td')
        
        # Move to next row if no columns
        if len(cols) != 0:        
            
            # Grab text
            filing_type = cols[0].text.strip()                 
            filing_date = cols[3].text.strip()
            filing_numb = cols[4].text.strip()
            
            # Grab links
            filing_doc_href = cols[1].find('a', {'href':True, 'id':'documentsbutton'})       
            filing_int_href = cols[1].find('a', {'href':True, 'id':'interactiveDataBtn'})
            filing_num_href = cols[4].find('a')
            
            # Grab first href
            if filing_doc_href != None:
                filing_doc_link = base_url_sec + filing_doc_href['href'] 
            else:
                filing_doc_link = 'no link'
            
            # Grab second href
            if filing_int_href != None:
                filing_int_link = base_url_sec + filing_int_href['href'] 
            else:
                filing_int_link = 'no link'
            
            # Grab third href
            if filing_num_href != None:
                filing_num_link = base_url_sec + filing_num_href['href'] 
            else:
                filing_num_link = 'no link'

            # End loop if enough data has been captured (11 years for annuals, 5 full years plus current FY for quarterly)
            if period == '10-k':
                if int(filing_date[:4]) + 11 < int(curYear):
                    break
            '''else: #for quarterly
                if date older:
                    break'''

            # Store data in dict
            file_dict = {}
            file_dict['file_type'] = filing_type
            file_dict['file_number'] = filing_numb
            file_dict['file_date'] = filing_date
            file_dict['links'] = {}
            file_dict['links']['documents'] = filing_doc_link[:-10] + '.txt'
            file_dict['links']['interactive_data'] = filing_int_link
            file_dict['links']['filing_number'] = filing_num_link
        
            # Add data to master list
            master_list.append(file_dict)

    print(master_list)
    

def quarterlyData(stock):
    pass

def annualData(stock):
    # Get filings
    filings = getFilings(stock.cik, '10-k')

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
    annualData(stock1)
    


    
#guiReturn = []
if __name__ == "__main__":
    main(guiReturn)


'''TODO
if filing is /A, skip next
if /A filing is last, skip
break condition for quarterly
pull quarter data from 10-k


'''