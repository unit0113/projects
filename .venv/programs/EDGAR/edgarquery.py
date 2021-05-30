import requests
import tkinter as tk
from tkinter import messagebox
from datetime import date
from datetime import datetime
from bs4 import BeautifulSoup
import pandas as pd
import time
import sys


class StockData:

    """ Class to store stock data for requested tickers, takes in initial input from edgargui
    """    
    
    def __init__(self, symbol, cik):
        """ Takes in ticker and CIK, then initiates the class

        Args:
            symbol (str): Stocker ticker
        """ 

        self.symbol = symbol
        self.cik = cik


def getFilings(cik, period, limit, headers):
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
    response = requests.get(url = endpoint, params = param_dict, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Build document table
    doc_table = soup.find_all('table', class_='tableFile2')

    # Base URL for link building
    base_url = r"https://www.sec.gov"

    # Loop through rows in table
    master_list = []
    today = date.today()
    curYear = today.year

    if type(limit) is not int:
        limit = datetime.strptime(limit, '%Y-%m-%d').date()

    for row in doc_table[0].find_all('tr'):
        
        # Find columns
        cols = row.find_all('td')
        
        # Move to next row if no columns
        if len(cols) != 0:        
            
            # Grab text
            filing_type = cols[0].text.strip()                 
            filing_date = cols[3].text.strip()
            filing_numb = cols[4].text.strip()

            # End loop if enough data has been captured (10 years for annuals, 5 full years plus current FY for quarterly)
            if type(limit) is int:
                if (int(filing_date[:4]) + limit) < int(curYear):
                    break
            else:
                if datetime.strptime(filing_date, '%Y-%m-%d').date() + pd.DateOffset(months=10) < limit:
                    break
            
            # Grab link
            filing_doc_href = cols[1].find('a', {'href':True, 'id':'documentsbutton'})       
            
            # Grab first href
            if filing_doc_href != None:
                filing_doc_link = base_url + filing_doc_href['href'] 
            else:
                filing_doc_link = 'no link'

            # Store data in dict
            file_dict = {}
            file_dict['file_type'] = filing_type
            file_dict['file_number'] = filing_numb
            file_dict['file_date'] = filing_date
            file_dict['documents'] = filing_doc_link[:-31].replace('-', '') + '/index.json'

            # Add data to master list
            master_list.append(file_dict)

    return master_list    


def quarterlyData(stock, headers, max_attempts = 5):
    # Get annual filings
    for i in range(max_attempts):
        try:
            filings = getFilings(stock.cik, '10-k', 6, headers)
        except:
            time.sleep(0.25)
            continue
        else:
            break
    else:
        tk.messagebox.showerror(title="Error", message="Network Error2. Please try again")
        sys.exit() 

    for i in range(max_attempts):
        try:
            filings.append(getFilings(stock.cik, '10-q', filings[-1]['file_date'], headers))
        except:
            time.sleep(0.25)
            continue
        else:
            break
    else:
        tk.messagebox.showerror(title="Error", message="Network Error3. Please try again")
        sys.exit()    
    
    return filings


def annualData(stock, headers, max_attempts = 5):
    # Get filings
    for i in range(max_attempts):
        try:
            filings = getFilings(stock.cik, '10-k', 11, headers)
        except:
            time.sleep(0.25)
            continue
        else:
            break
    else:
        tk.messagebox.showerror(title="Error", message="Network Error4. Please try again")
        sys.exit()
    return filings


def parse_filings(filings, stock, type, headers, max_attempts = 5):

    # Base url for link building
    base_url_sec = r"https://www.sec.gov"

    for filing in filings:
        for i in range(max_attempts):
            try:
                content = requests.get(filing['documents'], headers=headers).json()
            except:
                time.sleep(0.25)
                continue
            else:
                break
        else:
            tk.messagebox.showerror(title="Error", message="Network Error5. Please try again")
            sys.exit()
        
        for file in content['directory']['item']:  
            # Grab the filing summary and create a new url leading to the file so we can download it.
            if file['name'] == 'FilingSummary.xml':
                xml_summary = base_url_sec + content['directory']['name'] + "/" + file['name']

                # Define a new base url that represents the filing folder
                base_url = xml_summary.replace('FilingSummary.xml', '')

                # Request and parse the content
                time.sleep(.125)
                for j in range(max_attempts):
                    try:
                        content = requests.get(xml_summary, headers=headers).content
                        soup = BeautifulSoup(content, 'lxml')

                        # Find the 'myreports' tag because this contains all the individual reports submitted.
                        reports = soup.find('myreports')
                        temp = reports.find_all('report')[:-1]
                    except:
                        time.sleep(.25)
                        continue
                    else:
                        break
                else:
                    tk.messagebox.showerror(title="Error", message="Network Error6. Please try again")
                    sys.exit()

                master_reports = []
                # Loop through each report in the 'myreports' tag but avoid the last one as this will cause an error.
                for report in reports.find_all('report')[:-1]:
                    # Create a dictionary to store all the different parts we need.
                    report_dict = {}
                    report_dict['name_short'] = report.shortname.text
                    report_dict['name_long'] = report.longname.text
                    try:
                        report_dict['url'] = base_url + report.htmlfilename.text
                    except:
                        report_dict['url'] = base_url + report.xmlfilename.text[:-4] + '.htm'

                    # Append the dictionary to the master list.
                    master_reports.append(report_dict)
                break





def main(guiReturn, headers):
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
    headers = headers

    # Get list of fillings
    annual_filings = annualData(stock1, headers)    
    quarterly_filings = quarterlyData(stock1, headers)   

    # Pull data from filings
    parse_filings(annual_filings, stock1, 'annual', headers)


    
#guiReturn = []
if __name__ == "__main__":
    main(guiReturn, header)


'''TODO
if filing is /A, skip next
if /A filing is last, skip
skip next filing if previous is amended (period of report?)
pull quarter data from 10-k
complete comment strings
remove extra stuff from get filings (currently commented out)
'''