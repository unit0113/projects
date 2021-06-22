import requests
import tkinter as tk
from tkinter import messagebox
from datetime import date
from datetime import datetime
from bs4 import BeautifulSoup
import pandas as pd
import time
import sys
import re


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
    
    # Pull fiscal period from .htm
    def fy_html(soup):
        fy = '---'
        for row in soup.table.find_all('tr'):
            tds = row.find_all('td')
            if 'DocumentFiscalYearFocus' in str(tds):
                fy = int(''.join(re.findall(r'\d+', tds[1].text.strip())))
                return fy


    # Pull fiscal period from .htm
    def fy_xml(soup):
        fy = '---'
        rows = soup.find_all('Row')
        for row in rows:
            if 'DocumentFiscalYearFocus' in str(row):
                for cell in row:
                    for line in str(cell).split('\n'):
                        if 'NonNumbericText' in str(line):
                            fy = int(re.findall(r'\d\d\d\d', line.strip())[0])
                            return fy


    # Pull income data
    def rev_html(soup):
        rev = gross = oi = net = eps = '---'
        rev_check = 0
        cost_check = 0
        for row in soup.table.find_all('tr'):
            tds = row.find_all('td')
            if 'defref_us-gaap_Revenue' in str(tds) and rev_check == 0 or 'this, \'defref_us-gaap_SalesRevenueNet\', window' in str(tds) and rev_check == 0:
                rev = int(''.join(re.findall(r'\d+', tds[1].text.strip())))
                rev_check = 1
            elif 'defref_us-gaap_GrossProfit' in str(tds):
                gross = int(''.join(re.findall(r'\d+', tds[1].text.strip())))
            elif 'defref_us-gaap_OperatingIncomeLoss' in str(tds):
                oi = int(''.join(re.findall(r'\d+', tds[1].text.strip())))
            elif 'defref_us-gaap_NetIncomeLoss' in str(tds):
                net = int(''.join(re.findall(r'\d+', tds[1].text.strip())))
            elif 'EarningsPerShareDiluted' in str(tds):
                eps = float(''.join(re.findall(r'\d+\.\d+', tds[1].text.strip())))
            elif 'this, \'defref_us-gaap_CostOfRevenue\', window' in str(tds) and cost_check == 0 or 'this, \'defref_us-gaap_CostOfGoodsAndServicesSold\', window' in str(tds) and cost_check == 0:
                cost = int(''.join(re.findall(r'\d+', tds[1].text.strip())))

        # Calculate gross if not found
        if gross == '---':
            gross = rev - cost

        return rev, gross, oi, net, eps
    

    # Pull income data
    def rev_xml(soup):
        rev = gross = oi = net = eps = '---'
        rows = soup.find_all('Row')
        for row in rows:
            if 'us-gaap_Revenues' in str(row):
                for cell in row:
                    for line in str(cell).split('\n'):
                        if 'RoundedNumericAmount' in str(line):
                            rev = int(re.findall(r'\d+', line.strip())[0])
                            break
            elif 'us-gaap_CostOfRevenue' in str(row):
                for cell in row:
                    for line in str(cell).split('\n'):
                        if 'RoundedNumericAmount' in str(line):
                            cost = int(re.findall(r'\d+', line.strip())[0])
                            break                
                gross = rev - cost
            elif 'us-gaap_OperatingIncomeLoss' in str(row):
                for cell in row:
                    for line in str(cell).split('\n'):
                        if 'RoundedNumericAmount' in str(line):
                            oi = int(re.findall(r'\d+', line.strip())[0])
                            break 
            elif 'us-gaap_NetIncomeLoss' in str(row):
                for cell in row:
                    for line in str(cell).split('\n'):
                        if 'RoundedNumericAmount' in str(line):
                            net = int(re.findall(r'\d+', line.strip())[0])
                            break                
            elif 'us-gaap_EarningsPerShareDiluted' in str(row):
                for cell in row:
                    for line in str(cell).split('\n'):
                        if 'RoundedNumericAmount' in str(line):
                            eps = float(''.join(re.findall(r'\d+\.\d\d', line.strip())))
                            break
        
        return rev, gross, oi, net, eps 


    # Pull balance sheet data
    def bs_html(soup):
        cash = goodwill = assets = liabilities = '---'
        for row in soup.table.find_all('tr'):
            tds = row.find_all('td')
            if 'this, \'defref_us-gaap_CashAndCashEquivalentsAtCarryingValue\', window' in str(tds):
                for td in tds:
                    if 'nump' in str(td):
                        cash = int(''.join(re.findall(r'\d+', td.text.strip())))
                        break
            elif 'defref_us-gaap_Goodwill' in str(tds):
                for td in tds:
                    if 'nump' in str(td):
                        goodwill = int(''.join(re.findall(r'\d+', td.text.strip())))
                        break
            elif 'Total assets' in str(tds):
                for td in tds:
                    if 'nump' in str(td):
                        assets = int(''.join(re.findall(r'\d+', td.text.strip())))
                        break
            elif 'onclick="top.Show.showAR( this, \'defref_us-gaap_Liabilities\', window )' in str(tds):
                for td in tds:
                    if 'nump' in str(td):
                        liabilities = int(''.join(re.findall(r'\d+', td.text.strip())))
                        break
    
        # Remove goodwill from assets
        assets -= goodwill

        return cash, assets, liabilities

    # Pull cash flow data
    def cf_html(soup):
        cfo = capex = buyback = divpaid = '---'
        for row in soup.table.find_all('tr'):
            tds = row.find_all('td')
            if 'Net cash from operations' in str(tds):
                cfo = int(''.join(re.findall(r'\d+', tds[1].text.strip())))
            elif 'Additions to property and equipment' in str(tds):
                capex = int(''.join(re.findall(r'\d+', tds[1].text.strip())))
            elif 'Common stock repurchased' in str(tds):
                buyback = int(''.join(re.findall(r'\d+', tds[1].text.strip())))
            elif 'Common stock cash dividends paid' in str(tds):
                divpaid = int(''.join(re.findall(r'\d+', tds[1].text.strip())))
        
        # Calculate FCF
        try:
            fcf = cfo - capex
        except:
            fcf = '---'

        return fcf, buyback, divpaid

    # Pull div data
    def div_html(soup):
        div = '---'
        for row in soup.table.find_all('tr'):
            tds = row.find_all('td')
            if 'onclick="top.Show.showAR( this, \'defref_us-gaap_DividendsPayableDateDeclaredDayMonthAndYear\', window )' in str(tds):
                for index, row in enumerate(tds):
                    if not re.findall(r'\d', str(row)):
                        break
            elif 'onclick="top.Show.showAR( this, \'defref_us-gaap_CommonStockDividendsPerShareDeclared\', window )' in str(tds):
                div = float(''.join(re.findall(r'\d+\.\d+', tds[index].text.strip())))

        return div

    # Define statements to Parse
    intro_list = r'DOCUMENT AND ENTITY INFORMATION', 
    income_list = r'CONSOLIDATED STATEMENTS OF EARNINGS', r'STATEMENT OF INCOME ALTERNATIVE', r'CONSOLIDATED STATEMENT OF INCOME', r'INCOME STATEMENTS', r'STATEMENT OF INCOME', r'CONSOLIDATED STATEMENTS OF OPERATIONS', r'STATEMENTS OF CONSOLIDATED INCOME', r'CONSOLIDATED STATEMENTS OF INCOME', r'CONSOLIDATED STATEMENT OF OPERATIONS'
    bs_list = r'BALANCE SHEETS', r'CONSOLIDATED BALANCE SHEETS', r'STATEMENT OF FINANCIAL POSITION CLASSIFIED', r'CONSOLIDATED BALANCE SHEET'
    cf_list = r'CASH FLOWS STATEMENTS', r'CONSOLIDATED STATEMENTS OF CASH FLOWS', r'STATEMENT OF CASH FLOWS INDIRECT', r'CONSOLIDATED STATEMENT OF CASH FLOWS', r'STATEMENTS OF CONSOLIDATED CASH FLOWS'
    div_list = r'DIVIDENDS DECLARED (DETAIL)', 

    # Lists for data frame
    Fiscal_Period = []
    Revenue = []
    Gross_Profit = []
    Operating_Income = []
    Net_Profit = []
    Earnings_Per_Share = []
    Cash = []
    Total_Assets = []
    Total_Liabilities = []
    Free_Cash_Flow = []
    Share_Buybacks = []
    Dividend_Payments = []
    Dividends = []
    
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
            # Grab the filing summary url
            if file['name'] == 'FilingSummary.xml':
                xml_summary = r"https://www.sec.gov" + content['directory']['name'] + "/" + file['name']
                print(xml_summary)
                
                # Define a new base url
                base_url = xml_summary.replace('FilingSummary.xml', '')
                break

        # Request and parse the content
        time.sleep(.11)
        for attempt in range(max_attempts):
            try:
                content = requests.get(xml_summary, headers=headers).content
                soup = BeautifulSoup(content, 'lxml')

                # Find the all of the individual reports submitted
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
            
            # Summary table
            if report.shortname.text.upper() in intro_list:
                # Get URL and contents
                try:
                    intro_url = base_url + report.htmlfilename.text
                    content = requests.get(intro_url, headers=headers).content
                    soup = BeautifulSoup(content, 'html.parser')
                    fy = fy_html(soup)
                except:
                    intro_url = base_url + report.xmlfilename.text
                    content = requests.get(intro_url, headers=headers).content
                    soup = BeautifulSoup(content, 'xml')
                    fy = fy_xml(soup)
        
            # Income Statement
            if report.shortname.text.upper() in income_list:
                # Get URL and contents
                try:
                    rev_url = base_url + report.htmlfilename.text
                    content = requests.get(rev_url, headers=headers).content
                    soup = BeautifulSoup(content, 'html.parser')
                    rev, gross, oi, net, eps = rev_html(soup)
                except:
                    rev_url = base_url + report.xmlfilename.text
                    content = requests.get(rev_url, headers=headers).content
                    soup = BeautifulSoup(content, 'xml')
                    rev, gross, oi, net, eps = rev_xml(soup)

            # Balance sheet
            if report.shortname.text.upper() in bs_list:
                # Get URL and contents
                try:
                    bs_url = base_url + report.htmlfilename.text
                    content = requests.get(bs_url, headers=headers).content
                    soup = BeautifulSoup(content, 'html.parser')
                    cash, assets, liabilities = bs_html(soup)
                except:
                    bs_url = base_url + report.xmlfilename.text
                    content = requests.get(bs_url, headers=headers).content
                    soup = BeautifulSoup(content, 'xml')

            # Cash flow
            if report.shortname.text.upper() in cf_list:
                # Get URL and contents
                try:
                    cf_url = base_url + report.htmlfilename.text
                    content = requests.get(cf_url, headers=headers).content
                    soup = BeautifulSoup(content, 'html.parser')
                    fcf, buyback, divpaid = cf_html(soup)
                except:
                    cf_url = base_url + report.xmlfilename.text
                    content = requests.get(cf_url, headers=headers).content
                    soup = BeautifulSoup(content, 'xml')
            
             # Dividends
            if report.shortname.text.upper() in div_list:
                # Get URL and contents
                try:
                    cf_url = base_url + report.htmlfilename.text
                    content = requests.get(cf_url, headers=headers).content
                    soup = BeautifulSoup(content, 'html.parser')
                    div = div_html(soup)
                except:
                    cf_url = base_url + report.xmlfilename.text
                    content = requests.get(cf_url, headers=headers).content
                    soup = BeautifulSoup(content, 'xml')
        
        # Add parsed data to lists for data frame
        Fiscal_Period.append(fy)
        Revenue.append(rev)
        Gross_Profit.append(gross)
        Operating_Income.append(oi)
        Net_Profit.append(net)
        Earnings_Per_Share.append(eps)
        Cash.append(cash)
        Total_Assets.append(assets)
        Total_Liabilities.append(liabilities)
        Free_Cash_Flow.append(fcf)
        Share_Buybacks.append(buyback)
        Dividend_Payments.append(divpaid)
        Dividends.append(div)

        everything = [Fiscal_Period, Revenue, Gross_Profit, Operating_Income, Net_Profit, Earnings_Per_Share, Cash, Total_Assets, Total_Liabilities, Free_Cash_Flow, Share_Buybacks, Dividend_Payments, Dividends]
        print(everything)











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
put everything into the stock class
'''