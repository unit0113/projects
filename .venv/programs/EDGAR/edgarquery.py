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


def network_check_decorator(num, max_attempts=5):
    def inner(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    result = func(*args, **kwargs)
                except:
                    time.sleep(.25)
                    continue
                else:
                    break
            else:
                error_message = f"Network Error{num}. Please try again"
                tk.messagebox.showerror(title="Error", message=error_message)
                sys.exit()
            return result
        return wrapper
    return inner


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
            try:
                filings = getFilings(stock.cik, '10-k', 6, headers)
            except:
                filings = getFilings(stock.cik, '20-f', 6, headers)
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
            try:
                filings.append(getFilings(stock.cik, '10-q', filings[-1]['file_date'], headers))
            except:
                filings.append(getFilings(stock.cik, '20-f', filings[-1]['file_date'], headers))
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
            try:
                filings = getFilings(stock.cik, '10-k', 11, headers)
            except:
                filings = getFilings(stock.cik, '20-f', 11, headers)
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
    
    # Define statements to Parse
    item1 = r'INCOME STATEMENTS'
    item2 = r'BALANCE SHEETS'
    item3 = r'CASH FLOWS STATEMENTS'
    item4 = r'CONSOLIDATED STATEMENT OF OPERATIONS'
    item5 = r'DIVIDENDS DECLARED (DETAIL)'
    item6 = r'BASIC AND DILUTED EARNINGS PER SHARE (DETAIL)'
    item7 = r'CONSOLIDATED STATEMENTS OF CASH FLOWS'
    item8 = r'CONSOLIDATED STATEMENTS OF OPERATIONS'
    item9 = r'CONSOLIDATED BALANCE SHEETS'
    item10 = r'STATEMENT OF CASH FLOWS INDIRECT'
    item11 = r'STATEMENT OF INCOME'
    item12 = r'STATEMENT OF FINANCIAL POSITION CLASSIFIED'
    item13 = r'CONSOLIDATED BALANCE SHEET'
    item14 = r'CONSOLIDATED STATEMENT OF CASH FLOWS'
    item15 = r'CONSOLIDATED STATEMENTS OF EARNINGS'
    item16 = r'CONSOLIDATED STATEMENT OF INCOME'
    item17 = r'CONSOLIDATED STATEMENTS OF INCOME'
    item18 = r'STATEMENTS OF CONSOLIDATED INCOME'
    item19 = r'STATEMENTS OF CONSOLIDATED CASH FLOWS'
    item20 = r'STATEMENT OF INCOME ALTERNATIVE'
    report_list = [item1, item2, item3, item4, item5, item6, item7, item8, item9, item10, item11, item12, item13, item14, item15, item16, item17, item18, item19, item20]
    statements_url = []

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
                xml_summary = r"https://www.sec.gov" + content['directory']['name'] + "/" + file['name']
                
                # Define a new base url that represents the filing folder
                base_url = xml_summary.replace('FilingSummary.xml', '')

                # Request and parse the content
                time.sleep(.11)
                for attempt in range(max_attempts):
                    try:
                        content = requests.get(xml_summary, headers=headers).content
                        soup = BeautifulSoup(content, 'lxml')

                        # Find the 'myreports' tag because this contains all the individual reports submitted.
                        reports = soup.find('myreports')
                        test = reports.find_all('report')[:-1]
                    except:
                        time.sleep(.25)
                        continue
                    else:
                        break
                else:
                    tk.messagebox.showerror(title="Error", message="Network Error6. Please try again")
                    sys.exit()
               
                # Loop through each report with the 'myreports' tag but avoid the last one as this will cause an error
                for report in reports.find_all('report')[:-1]:
                    if report.shortname.text.upper() in report_list:
                        # Add Statement URL to List
                        try:
                            statements_url.append(base_url + report.htmlfilename.text)
                        except:
                            statements_url.append(base_url + report.xmlfilename.text)
                break

    # Create Data Set
    statements_data = []

    # Loop through statement URL's
    for statement in statements_url:

        # Dict to Store Statement
        statement_data = {}
        statement_data['headers'] = []
        statement_data['sections'] = []
        statement_data['data'] = []
        
        # Get Statement Data
        content = requests.get(statement).content
        report_soup = BeautifulSoup(content, 'html')

        # Loop Through Rows in Content
        for index, row in enumerate(report_soup.table.find_all('tr')):
            
            # Get All Elements
            cols = row.find_all('td')
            
            # Check if Regular Row
            if (len(row.find_all('th')) == 0 and len(row.find_all('strong')) == 0): 
                reg_row = [ele.text.strip() for ele in cols]
                statement_data['data'].append(reg_row)
                
            # Check if Section
            elif (len(row.find_all('th')) == 0 and len(row.find_all('strong')) != 0):
                sec_row = cols[0].text.strip()
                statement_data['sections'].append(sec_row)
                
            # Check if Header
            elif (len(row.find_all('th')) != 0):            
                hed_row = [ele.text.strip() for ele in row.find_all('th')]
                statement_data['headers'].append(hed_row)
                
            else:            
                print('We encountered an error.')

        # Append to Statements Data
        statements_data.append(statement_data)





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
            stock2_flag = True            
    excel_flag = guiReturn[4]
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