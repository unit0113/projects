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
        # Get list of fillings
        self.annual_filings = annualData(self.cik, headers)   
        #quarterly_filings = quarterlyData(self, headers)   

        # Pull data from filings
        parse_filings(self.annual_filings, 'annual', headers)


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

            # Check for amended filing
            if 'A' in filing_type:
                continue
            
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


@network_check_decorator(2)
def quarterlyData(cik, headers):
    # Get quarterly filings
    try:
        filings = getFilings(cik, '10-k', 6, headers)
        filings.append(getFilings(cik, '10-q', filings[-1]['file_date'], headers))
    except:
        filings = getFilings(cik, '20-f', 6, headers)
        filings.append(getFilings(cik, '20-f', filings[-1]['file_date'], headers))  
    
    return filings

@network_check_decorator(3)
def annualData(cik, headers):
    # Get filings
    try:
        filings = getFilings(cik, '10-k', 11, headers)
    except:
        filings = getFilings(cik, '20-f', 11, headers)    
    
    return filings


def parse_filings(filings, type, headers):    
    # Pull fiscal period from .htm
    def fy_html(soup):
        fy = period_end = '---'
        for row in soup.table.find_all('tr'):
            tds = row.find_all('td')
            if 'DocumentFiscalYearFocus' in str(tds):
                fy = int(tds[1].text)
            if 'this, \'defref_dei_DocumentPeriodEndDate\', window' in str(tds):
                period_end_str = tds[1].text.strip()
                period_end = datetime.strptime(period_end_str, '%b. %d,  %Y') 
        
            if fy != '---' and period_end != '---':
                return fy, period_end
        return fy, period_end


    # Pull fiscal period from .xml
    def fy_xml(soup):
        fy = '---'
        rows = soup.find_all('Row')
        for row in rows:
            if 'DocumentFiscalYearFocus' in str(row):
                for cell in row:
                    for line in str(cell).split('\n'):
                        if 'NonNumbericText' in str(line):
                            fy = int(re.findall(r'\d\d\d\d', line.strip())[0])
                            break
            if 'dei_DocumentPeriodEndDate' in str(row):
                for cell in row:
                    for line in str(cell).split('\n'):
                        if 'NonNumbericText' in str(line):
                            period_end_str = re.findall(r'<NonNumbericText>(.*?)</NonNumbericText>', line.strip())[0]
                            period_end = datetime.strptime(period_end_str, '%Y-%m-%d') 
                            break                  
                
        return fy, period_end


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
            elif 'this, \'defref_us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding\', window' in str(tds):
                shares = int(''.join(re.findall(r'\d+', tds[1].text.strip())))

        # Calculate gross if not found
        if gross == '---':
            gross = rev - cost

        return rev, gross, oi, net, eps, shares
    

    # Pull income data
    def rev_xml(soup):
        rev = gross = oi = net = eps = '---'
        rows = soup.find_all('Row')
        for row in rows:
            if 'us-gaap_Revenues' in str(row) or 'us-gaap_SalesRevenueNet' in str(row):
                for cell in row:
                    for line in str(cell).split('\n'):
                        if 'RoundedNumericAmount' in str(line):
                            rev_str = re.findall(r'<RoundedNumericAmount>(.*?)</RoundedNumericAmount>', line.strip())[0]
                            rev = int(re.findall(r'\d+', rev_str.strip())[0])
                            break
            elif 'us-gaap_CostOfRevenue' in str(row) or 'us-gaap_CostOfGoodsAndServicesSold' in str(row):
                for cell in row:
                    for line in str(cell).split('\n'):
                        if 'RoundedNumericAmount' in str(line):
                            cost_str = re.findall(r'<RoundedNumericAmount>(.*?)</RoundedNumericAmount>', line.strip())[0]
                            cost = int(re.findall(r'\d+', cost_str.strip())[0])
                            break                
                gross = rev - cost
            elif 'us-gaap_OperatingIncomeLoss' in str(row):
                for cell in row:
                    for line in str(cell).split('\n'):
                        if 'RoundedNumericAmount' in str(line):
                            oi_str = re.findall(r'<RoundedNumericAmount>(.*?)</RoundedNumericAmount>', line.strip())[0]
                            oi = int(re.findall(r'\d+', oi_str.strip())[0])
                            break      
            elif 'us-gaap_NetIncomeLoss' in str(row):
                for cell in row:
                    for line in str(cell).split('\n'):
                        if 'RoundedNumericAmount' in str(line):
                            net_str = re.findall(r'<RoundedNumericAmount>(.*?)</RoundedNumericAmount>', line.strip())[0]
                            net = int(re.findall(r'\d+', net_str.strip())[0])
                            break                
            elif 'us-gaap_EarningsPerShareDiluted' in str(row):
                for cell in row:
                    for line in str(cell).split('\n'):
                        if 'RoundedNumericAmount' in str(line):
                            eps_str = re.findall(r'<RoundedNumericAmount>(.*?)</RoundedNumericAmount>', line.strip())[0]
                            eps = float(''.join(re.findall(r'\d+\.\d\d', eps_str.strip())))
                            break
            elif 'us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding' in str(row):
                for cell in row:
                    for line in str(cell).split('\n'):
                        if 'RoundedNumericAmount' in str(line):
                            shares_str = re.findall(r'<RoundedNumericAmount>(.*?)</RoundedNumericAmount>', line.strip())[0]
                            shares = int(re.findall(r'\d+', shares_str.strip())[0])
                            break
        
        return rev, gross, oi, net, eps, shares 


    # Pull balance sheet data
    def bs_html(soup):
        cash = assets = liabilities = '---'
        goodwill = 0
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
            elif 'this, \'defref_us-gaap_LiabilitiesAndStockholdersEquity\', window' in str(tds):
                for td in tds:
                    if 'nump' in str(td):
                        tot_liabilities = int(''.join(re.findall(r'\d+', td.text.strip())))
                        break
            elif 'this, \'defref_us-gaap_StockholdersEquity\', window' in str(tds):
                for td in tds:
                    if 'nump' in str(td):
                        equity = int(''.join(re.findall(r'\d+', td.text.strip())))
                        break            

        if liabilities == '---':
            liabilities = tot_liabilities - equity

        # Remove goodwill from assets
        assets -= goodwill

        return cash, assets, liabilities

    # Pull balance sheet data
    def bs_xml(soup):
        cash = goodwill = assets = '---'
        rows = soup.find_all('Row')
        liabilities = 0 

        for row in rows:
            if 'us-gaap_CashCashEquivalentsAndShortTermInvestments' in str(row) or 'us-gaap_CashAndCashEquivalentsAtCarryingValue' in str(row):
                for cell in row:
                    for line in str(cell).split('\n'):
                        if 'RoundedNumericAmount' in str(line):
                            cash_str = re.findall(r'<RoundedNumericAmount>(.*?)</RoundedNumericAmount>', line.strip())[0]
                            cash = int(re.findall(r'\d+', cash_str.strip())[0])
                            break
            elif 'us-gaap_Goodwill' in str(row):
                for cell in row:
                    for line in str(cell).split('\n'):
                        if 'RoundedNumericAmount' in str(line):
                            goodwill_str = re.findall(r'<RoundedNumericAmount>(.*?)</RoundedNumericAmount>', line.strip())[0]
                            goodwill = int(re.findall(r'\d+', goodwill_str.strip())[0])
                            break                
            elif r'<ElementName>us-gaap_Assets</ElementName>' in str(row):
                for cell in row:
                    for line in str(cell).split('\n'):
                        if 'RoundedNumericAmount' in str(line):
                            assets_str = re.findall(r'<RoundedNumericAmount>(.*?)</RoundedNumericAmount>', line.strip())[0]
                            assets = int(re.findall(r'\d+', assets_str.strip())[0])
                            assets -= goodwill
                            break      
            elif r'<ElementName>us-gaap_LiabilitiesCurrent</ElementName>' in str(row):
                for cell in row:
                    for line in str(cell).split('\n'):
                        if 'RoundedNumericAmount' in str(line):
                            liabil_str = re.findall(r'<RoundedNumericAmount>(.*?)</RoundedNumericAmount>', line.strip())[0]
                            liabilities_current = int(re.findall(r'\d+', liabil_str.strip())[0])
                            liabilities += liabilities_current
                            break
            elif 'us-gaap_LongTermDebtNoncurrent' in str(row):
                for cell in row:
                    for line in str(cell).split('\n'):
                        if 'RoundedNumericAmount' in str(line):
                            debt_str = re.findall(r'<RoundedNumericAmount>(.*?)</RoundedNumericAmount>', line.strip())[0]
                            long_term_debt = int(re.findall(r'\d+', debt_str.strip())[0])
                            liabilities += long_term_debt
                            break
            elif 'us-gaap_OtherLiabilitiesNoncurrent' in str(row):
                for cell in row:
                    for line in str(cell).split('\n'):
                        if 'RoundedNumericAmount' in str(line):
                            liabil2_str = re.findall(r'<RoundedNumericAmount>(.*?)</RoundedNumericAmount>', line.strip())[0]
                            liabilities_other = int(re.findall(r'\d+', liabil2_str.strip())[0])
                            liabilities += liabilities_other
                            break

        return cash, assets, liabilities
        

    # Pull cash flow data
    def cf_html(soup):
        cfo = capex = buyback = divpaid = '---'
        debt = 0
        for row in soup.table.find_all('tr'):
            tds = row.find_all('td')
            if 'Net cash from operations' in str(tds) or 'this, \'defref_us-gaap_NetCashProvidedByUsedInOperatingActivities\', window' in str(tds) or 'this, \'defref_us-gaap_NetCashProvidedByUsedInOperatingActivitiesContinuingOperations\', window' in str(tds):
                cfo = int(''.join(re.findall(r'\d+', tds[1].text.strip())))
            elif 'Additions to property and equipment' in str(tds) or 'this, \'defref_us-gaap_PaymentsToAcquireProductiveAssets\', window' in str(tds) or'this, \'defref_us-gaap_PaymentsToAcquirePropertyPlantAndEquipment\', window' in str(tds) or 'this, \'defref_us-gaap_PaymentsForProceedsFromProductiveAssets\', window' in str(tds):
                capex = int(''.join(re.findall(r'\d+', tds[1].text.strip())))
            elif 'this, \'defref_us-gaap_ProceedsFromRepaymentsOfShortTermDebtMaturingInThreeMonthsOrLess\', window' in str(tds) or 'this, \'defref_us-gaap_RepaymentsOfDebtMaturingInMoreThanThreeMonths\', window' in str(tds) or 'this, \'defref_us-gaap_RepaymentsOfLongTermDebt\', window' in str(tds):
                debt += int(''.join(re.findall(r'\d+', tds[1].text.strip())))
            elif 'Common stock repurchased' in str(tds) or 'this, \'defref_us-gaap_PaymentsForRepurchaseOfCommonStock\', window' in str(tds):
                buyback = int(''.join(re.findall(r'\d+', tds[1].text.strip())))
            elif 'Common stock cash dividends paid' in str(tds) or 'this, \'defref_us-gaap_PaymentsOfDividends\', window' in str(tds) or 'PaymentsOfDividendsAndDividendEquivalentsOnCommonStockAndRestrictedStockUnits\', window' in str(tds):
                divpaid = int(''.join(re.findall(r'\d+', tds[1].text.strip())))
        
        # Calculate FCF
        try:
            fcf = cfo - capex - debt
        except:
            fcf = '---'

        return fcf, buyback, divpaid

    def cf_xml(soup):
        cfo = capex = buyback = divpaid = '---'
        debt = 0
        rows = soup.find_all('Row')
        for row in rows:
            if 'us-gaap_NetCashProvidedByUsedInOperatingActivities' in str(row):
                for cell in row:
                    for line in str(cell).split('\n'):
                        if 'RoundedNumericAmount' in str(line):
                            cfo_str = re.findall(r'<RoundedNumericAmount>(.*?)</RoundedNumericAmount>', line.strip())[0]
                            cfo = int(re.findall(r'\d+', cfo_str.strip())[0])
                            break
            elif 'us-gaap_PaymentsToAcquirePropertyPlantAndEquipment' in str(row) or 'us-gaap_PaymentsToAcquireProductiveAssets' in str(row):
                for cell in row:
                    for line in str(cell).split('\n'):
                        if 'RoundedNumericAmount' in str(line):
                            capex_str = re.findall(r'<RoundedNumericAmount>(.*?)</RoundedNumericAmount>', line.strip())[0]
                            capex = int(re.findall(r'\d+', capex_str.strip())[0])
                            break
            elif 'us-gaap_ProceedsFromRepaymentsOfShortTermDebtMaturingInThreeMonthsOrLess' in str(row) or 'RepaymentsOfShortTermAndLongTermBorrowings' in str(row) or 'us-gaap_RepaymentsOfLongTermDebtAndCapitalSecurities' in str(row):
                for cell in row:
                    for line in str(cell).split('\n'):
                        if 'RoundedNumericAmount' in str(line):
                            debt_str = re.findall(r'<RoundedNumericAmount>(.*?)</RoundedNumericAmount>', line.strip())[0]
                            debt += int(re.findall(r'\d+', debt_str.strip())[0])
                            break                                
            elif 'us-gaap_PaymentsForRepurchaseOfCommonStock' in str(row):
                for cell in row:
                    for line in str(cell).split('\n'):
                        if 'RoundedNumericAmount' in str(line):
                            buyback_str = re.findall(r'<RoundedNumericAmount>(.*?)</RoundedNumericAmount>', line.strip())[0]
                            buyback = int(re.findall(r'\d+', buyback_str.strip())[0])
                            break      
            elif 'us-gaap_PaymentsOfDividendsCommonStock' in str(row):
                for cell in row:
                    for line in str(cell).split('\n'):
                        if 'RoundedNumericAmount' in str(line):
                            divpaid_str = re.findall(r'<RoundedNumericAmount>(.*?)</RoundedNumericAmount>', line.strip())[0]
                            divpaid = int(re.findall(r'\d+', divpaid_str.strip())[0])
                            break
        
        # Calculate FCF
        try:
            fcf = cfo - capex - debt
        except:
            fcf = '---'

        return fcf, buyback, divpaid

    # Pull div data
    def div_html(soup):
        div = '---'    
        # If company has seperate div table
        if soup.find('th').text in div_list[:1]:
            for row in soup.table.find_all('tr'):
                tds = row.find_all('td')
                if 'onclick="top.Show.showAR( this, \'defref_us-gaap_DividendsPayableDateDeclaredDayMonthAndYear\', window )' in str(tds):
                    for index, row in enumerate(tds):
                        if not re.findall(r'\d', str(row)):
                            break
                elif 'onclick="top.Show.showAR( this, \'defref_us-gaap_CommonStockDividendsPerShareDeclared\', window )' in str(tds):
                    div = float(''.join(re.findall(r'\d+\.\d+', tds[index].text.strip())))
                    break

        # If company is a jerk and burries it (like apple)
        else:
            # Find row where 12 month data starts
            index = 1
            for header in soup.table.find_all('th'):
                if 'Months Ended' in str(header) and 'colspan' in str(header) and '12 Months Ended' not in str(header):
                    index_str = re.findall(r'\d', str(header))
                    index += int(index_str[0])

                if '12 Months Ended' in str(header):
                    break

            for row in soup.table.find_all('tr'):
                tds = row.find_all('td')
                if 'this, \'defref_us-gaap_CommonStockDividendsPerShareDeclared\', window' in str(tds):
                    try:
                        div = float(''.join(re.findall(r'\d+\.\d+', tds[index].text.strip())))
                    except:
                        for td in tds:
                            if 'nump' in str(td):
                                div = float(''.join(re.findall(r'\d+\.\d+', td.text.strip())))
                                break
                    break
            
            if div == '---':
                tables = soup.find_all('table')[0] 
                panda = pd.read_html(str(tables))
                for table in panda:
                    if 'Dividends Per Share' in table.values or 'DividendsPer Share' in table.values:
                        try:
                            row = table.loc[table[0] == 'Total cash dividends declared and paid']
                            div = float(re.findall(r'\d+\.\d\d', str(row))[0])
                        except:
                            row = table.loc[table[0] == 'Total']
                            div = float(re.findall(r'\d+\.\d\d', str(row))[0])
                        break

        return div


    # Pull div data
    def div_xml(soup):
        div = '---'
        
        # Find which cell yearly data starts in
        cols = soup.find_all('Column')
        for index, col in enumerate(cols):
            if '12 Months Ended' in str(col):
                break
        search_id = '<Id>' + str(index + 1) + r'</Id>'

        rows = soup.find_all('Row')
        for row in rows:
            if '>us-gaap_CommonStockDividendsPerShareDeclared' in str(row):
                cells = row.find_all('Cell')
                for cell in cells:
                    if search_id in str(cell):
                        for line in str(cell).split('\n'):
                            if 'NumericAmount' in str(line):
                                div_str = re.findall(r'<NumericAmount>(.*?)</NumericAmount>', line.strip())[0]
                                div = float(re.findall(r'\d+\.\d+', div_str.strip())[0])
                                break
                        break

        return div

    # Define statements to Parse
    intro_list = ['DOCUMENT AND ENTITY INFORMATION', 'COVER PAGE']
    income_list = ['CONSOLIDATED STATEMENTS OF EARNINGS', 'STATEMENT OF INCOME ALTERNATIVE', 'CONSOLIDATED STATEMENT OF INCOME', 'INCOME STATEMENTS', 'STATEMENT OF INCOME',
                   'CONSOLIDATED STATEMENTS OF OPERATIONS', 'STATEMENTS OF CONSOLIDATED INCOME', 'CONSOLIDATED STATEMENTS OF INCOME', 'CONSOLIDATED STATEMENT OF OPERATIONS']
    bs_list = ['BALANCE SHEETS', 'CONSOLIDATED BALANCE SHEETS', 'STATEMENT OF FINANCIAL POSITION CLASSIFIED', 'CONSOLIDATED BALANCE SHEET']
    cf_list = ['CASH FLOWS STATEMENTS', 'CONSOLIDATED STATEMENTS OF CASH FLOWS', 'STATEMENT OF CASH FLOWS INDIRECT', 'CONSOLIDATED STATEMENT OF CASH FLOWS',
               'STATEMENTS OF CONSOLIDATED CASH FLOWS']
    div_list = ['DIVIDENDS DECLARED (DETAIL)', 'CONSOLIDATED STATEMENTS OF SHAREHOLDERS\' EQUITY', 'CONSOLIDATED STATEMENTS OF SHAREHOLDERS\' EQUITY CONSOLIDATED STATEMENTS OF SHAREHOLDERS\' EQUITY (PARENTHETICAL)',
                'SHAREHOLDERS\' EQUITY', 'SHAREHOLDERS\' EQUITY AND SHARE-BASED COMPENSATION - ADDITIONAL INFORMATION (DETAIL) (USD $)', 'SHAREHOLDERS\' EQUITY - ADDITIONAL INFORMATION (DETAIL)',
                'SHAREHOLDERS\' EQUITY AND SHARE-BASED COMPENSATION - ADDITIONAL INFORMATION (DETAIL)']

    # Lists for data frame
    Fiscal_Period = []
    Period_End = []
    Revenue = []
    Gross_Profit = []
    Operating_Income = []
    Net_Profit = []
    Earnings_Per_Share = []
    Cash = []
    Total_Assets = []
    Total_Liabilities = []
    Shares_Outstanding = []
    Free_Cash_Flow = []
    Share_Buybacks = []
    Dividend_Payments = []
    Dividends = []
    

    @network_check_decorator(4)
    def pull_filing(filing):
        content = requests.get(filing['documents'], headers=headers).json()
        return content

    @network_check_decorator(5)
    def pull_filing_2(xml_summary):
        content = requests.get(xml_summary, headers=headers).content
        soup = BeautifulSoup(content, 'lxml')

        # Find the all of the individual reports submitted
        reports = soup.find('myreports')
        assert reports != None
        return reports

    for filing in filings:
        content = pull_filing(filing)
        print(filing)

        for file in content['directory']['item']:  
            # Grab the filing summary url
            if file['name'] == 'FilingSummary.xml':
                xml_summary = r"https://www.sec.gov" + content['directory']['name'] + "/" + file['name']
                print(xml_summary)
                
                # Define a new base url
                base_url = xml_summary.replace('FilingSummary.xml', '')
                break

        reports = pull_filing_2(xml_summary)
        div = '---'

        # Loop through each report with the 'myreports' tag but avoid the last one as this will cause an error
        for report in reports.find_all('report')[:-1]:
            
            # Summary table
            if report.shortname.text.upper() in intro_list:
                # Get URL and contents
                try:
                    intro_url = base_url + report.htmlfilename.text
                    content = requests.get(intro_url, headers=headers).content
                    soup = BeautifulSoup(content, 'html.parser')
                    fy, period_end = fy_html(soup)
                except:
                    intro_url = base_url + report.xmlfilename.text
                    content = requests.get(intro_url, headers=headers).content
                    soup = BeautifulSoup(content, 'xml')
                    fy, period_end = fy_xml(soup)
        
            # Income Statement
            if report.shortname.text.upper() in income_list:
                # Get URL and contents
                try:
                    rev_url = base_url + report.htmlfilename.text
                    content = requests.get(rev_url, headers=headers).content
                    soup = BeautifulSoup(content, 'html.parser')
                    rev, gross, oi, net, eps, shares = rev_html(soup)
                except:
                    rev_url = base_url + report.xmlfilename.text
                    content = requests.get(rev_url, headers=headers).content
                    soup = BeautifulSoup(content, 'xml')
                    rev, gross, oi, net, eps, shares = rev_xml(soup)

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
                    cash, assets, liabilities = bs_xml(soup)

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
                    fcf, buyback, divpaid = cf_xml(soup)
            
            # Dividends
            if report.shortname.text.upper() in div_list and div == '---':
                # Get URL and contents
                try:
                    div_url = base_url + report.htmlfilename.text
                    content = requests.get(div_url, headers=headers).content
                    soup = BeautifulSoup(content, 'html.parser')
                    div = div_html(soup)
                except:
                    div_url = base_url + report.xmlfilename.text
                    content = requests.get(div_url, headers=headers).content
                    soup = BeautifulSoup(content, 'xml')
                    div = div_xml(soup)


        # Check if FY same as previous year
        if len(Fiscal_Period) > 0:
            if fy == Fiscal_Period[-1]:
                continue

        # Add parsed data to lists for data frame
        Fiscal_Period.append(fy)
        Period_End.append(period_end)
        Revenue.append(rev)
        Gross_Profit.append(gross)
        Operating_Income.append(oi)
        Net_Profit.append(net)
        Earnings_Per_Share.append(eps)
        Cash.append(cash)
        Total_Assets.append(assets)
        Total_Liabilities.append(liabilities)
        Shares_Outstanding.append(shares)
        Free_Cash_Flow.append(fcf)
        Share_Buybacks.append(buyback)
        Dividend_Payments.append(divpaid)
        Dividends.append(div)

        everything = {'FY': Fiscal_Period, 'Per': Period_End, 'Rev': Revenue, 'Gross': Gross_Profit, 'OI': Operating_Income, 'Net': Net_Profit, 'EPS': Earnings_Per_Share, 'Cash': Cash,
                      'Assets': Total_Assets, 'Liabilities': Total_Liabilities, 'Shares': Shares_Outstanding, 'FCF': Free_Cash_Flow, 'Buybacks': Share_Buybacks,
                      'Div_Paid': Dividend_Payments, 'Div': Dividends}
        print(everything)
        print('-'*50)
    
    # Determine if share split occured, and adjust per share metrics
    split_factor = 1
    
    # Loop through share count and look for major differences
    for i in range(1, len(Shares_Outstanding)):
        share_ratio = round(Shares_Outstanding[i-1] / Shares_Outstanding[i])
        
        # If difference found, update split factor
        if share_ratio >= 1.25 or share_ratio <= 0.75:
            split_factor *= share_ratio
        
        # If there have been splits, adjust per share metrics
        if split_factor != 1:
            Earnings_Per_Share[i] = round(Earnings_Per_Share[i] / split_factor, 2)
            if Dividends[i] != '---':
                Dividends[i] = round(Dividends[i] / split_factor, 2)











def main(gui_return, header):
    """ Runs two subprograms. Webscrapes ticker/CIK paires from SEC site, gets tickers from user and passes the CIK codes and a seperate flag to the edgarquery program

    Returns:
        cik1 (str): first CIK input
        cik2 (str): second CIK input
        excelFlag (bool): excel export flag
    """
    # Pull data from GUI and initiate StockData class(es)
    excel_flag = gui_return[1]
    global headers
    headers = header
    stocks = []
    for item in gui_return[0]:
        stocks.append(StockData(item[0], item[1]))

    
if __name__ == "__main__":
    main(gui_return, header)


'''TODO
pull quarter data from 10-k
complete comment strings
'''