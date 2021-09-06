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
import math


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
    #amended_check = False

    if type(limit) is not int:
        limit = datetime.strptime(limit, '%Y-%m-%d').date()

    for row in doc_table[0].find_all('tr'):
        
        '''# Check if previous was amended
        if amended_check == True:
            amended_check = False
            continue'''

        # Find columns
        cols = row.find_all('td')
        
        # Move to next row if no columns
        if len(cols) != 0:        
            
            # Grab text
            filing_type = cols[0].text.strip()                 
            filing_date = cols[3].text.strip()
            filing_numb = cols[4].text.strip()

            '''# Check for amended filing
            if 'A' in filing_type:
                    amended_check = True'''
            
            if 'A' in filing_type:
                continue

            # End loop if enough data has been captured (default 10 years for annuals, 5 full years plus current FY for quarterly)
            if type(limit) is int:
                if len(master_list) > (limit):
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

    return master_list[:limit]    


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
    ''' Parse filings and create datafram from the data

    Args:
        filings (lst of dicts contains filings data), type (str, options of annual or quarterly), headers (dict, user agent for accessing SEC website w/o getting yelled at)
    Return:
        results (dataframe)       
    '''
    
    # RE for .htm filings
    def html_re(string):
        ''' RE syntax and error checking for pulling numeric information from data cell on .htm filings

        Args:
            Target string (str)
        Return:
            results (int/float)       
        '''

        # Pull numeric data from string
        obj = re.search(r'(?:<td class=\"(?:num|nump)\">)(?:.*?\(?)(\d*,?\d*,?.?\d+)(?:\)?<span>)', string, re.M)
        try:
            result = obj.group(1).replace(',', '')
        except:
            return '---'
        
        # Looks for blank cell to ensure that correct data is returned
        empty = re.search(r'text', string, re.M)
        if empty is not None:
            if empty.span()[0] < obj.span()[0]:
                return '---'
        
        # Check if return value is float or int
        if '.' in result:
            return float(result)
        else:
            return int(result)


    # RE for .xml filings
    def xml_re(string):
        ''' RE syntax and error checking for pulling numeric information from data cell on .xml filings

        Args:
            Target string (str)
        Return:
            results (int/float)       
        '''

        # Pull numeric data from string
        obj = re.search(r'(?:<RoundedNumericAmount>-?)(.*?)(?:</RoundedNumericAmount>)', string, re.M)
        try:
            result = obj.group(1).replace(',', '')
        except:
            return '---'
        
        # Looks for blank cell to ensure that correct data is returned
        empty = re.search(r'text', string, re.M)
        if empty is not None:
            if empty.span()[0] < obj.span()[0]:
                return '---'
        
        # Check if return value is float or int
        if '.' in result:
            return float(result)
        else:
            return int(result)
    
    
    # Pull fiscal period from .htm
    def fy_html(soup):
        ''' Parse .htm document information

        Args:
            soup (soup data of sub filings)
        Returns:
            Fiscal year (int), Fiscal period end date (datetime object)        
        '''
        
        # Initial values
        fy = period_end = '---'
        
        # Loop through rows, search for FY and period end date
        for row in soup.table.find_all('tr'):
            tds = row.find_all('td')
            if 'DocumentFiscalYearFocus' in str(tds):
                fy = int(tds[1].text)
            if 'this, \'defref_dei_DocumentPeriodEndDate\', window' in str(tds):
                period_end_str = tds[1].text.strip()
                period_end = datetime.strptime(period_end_str, '%b. %d,  %Y') 
        
        return fy, period_end


    # Pull fiscal period from .xml
    def fy_xml(soup):
        ''' Parse .xml document information

        Args:
            soup (soup data of sub filings)
        Returns:
            Fiscal year (int), Fiscal period end date (datetime object)        
        '''

        # Initial values
        fy = period_end = '---'
        
        # Loop through rows, search for FY and period end date
        rows = soup.find_all('Row')
        for row in rows:
            if 'DocumentFiscalYearFocus' in str(row):
                obj = re.search(r'(?:<NonNumbericText>)(\d+)(?:</NonNumbericText>)', str(row), re.M)
                fy = int(obj.group(1))
            elif 'dei_DocumentPeriodEndDate' in str(row):
                obj = re.search(r'(?:<NonNumbericText>)(.*?)(?:</NonNumbericText>)', str(row), re.M)
                period_end = datetime.strptime(obj.group(1), '%Y-%m-%d')                  
                
        return fy, period_end


    # Pull income data
    def rev_html(soup):
        ''' Parse .htm income statement

        Args:
            soup (soup data of sub filings)
        Returns:
            revenue (int), gross profit (int), R&D costs (int), operating income (int), net income (int), earnings per share (float), share count (int), dividend (float)      
        '''

        # Initial values
        rev = gross = oi = net = eps = cost = shares = div = '---'
        share_sum = research = 0

        # Loop through rows, search for row of interest
        for row in soup.table.find_all('tr'):
            tds = row.find_all('td')
            if ('this, \'defref_us-gaap_Revenues\', window' in str(tds) and rev == '---' or
                'this, \'defref_us-gaap_SalesRevenueNet\', window' in str(tds) and rev == '---' or
                'defref_us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax' in str(tds) and rev == '---'
                ):
                rev = html_re(str(tds))
            elif 'defref_us-gaap_GrossProfit' in str(tds):
                gross = html_re(str(tds))
            elif ('this, \'defref_us-gaap_ResearchAndDevelopmentExpense\', window' in str(tds) or
                'this, \'defref_amzn_TechnologyAndContentExpense\', window' in str(tds)
                ):
                research = html_re(str(tds))
            elif 'this, \'defref_us-gaap_OperatingIncomeLoss\', window' in str(tds):
                oi = html_re(str(tds))
            elif ('this, \'defref_us-gaap_NetIncomeLoss\', window' in str(tds) and net == '---' or
                'this, \'defref_us-gaap_ProfitLoss\', window );">Net income' in str(tds) and net == '---'
                ):
                net = html_re(str(tds))
            elif 'EarningsPerShareDiluted' in str(tds) and eps == '---':
                eps = html_re(str(tds))
            elif ('this, \'defref_us-gaap_CostOfRevenue\', window' in str(tds) and cost == '---' or
                'this, \'defref_us-gaap_CostOfGoodsAndServicesSold\', window' in str(tds) and cost == '---'
                ):
                cost = html_re(str(tds))
            elif 'this, \'defref_us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding\', window' in str(tds):
                shares = html_re(str(tds))
                share_sum += shares
            elif 'this, \'defref_us-gaap_CommonStockDividendsPerShareDeclared\', window' in str(tds):
                div = html_re(str(tds))
                
        # Calculate gross if not listed
        if gross == '---':
            try:
                gross = rev - cost
            except:
                gross = rev

        # Calculate shares if not listed
        if share_sum == 0 and eps != '---':
            share_sum = round(net / eps)

        # Calculate EPS if not listed
        if eps == '---':
            eps = net / share_sum

        return rev, gross, research, oi, net, eps, share_sum, div
    

    # Pull income data
    def rev_xml(soup):
        ''' Parse .xml income statement

        Args:
            soup (soup data of sub filings)
        Returns:
            revenue (int), gross profit (int), R&D costs (int), operating income (int), net income (int), earnings per share (float), share count (int), dividend (float)      
        '''    
        
        # Initial values
        rev = gross = oi = net = eps = cost = shares = div = '---'
        share_sum = research = 0

        # Loop through rows, search for row of interest
        rows = soup.find_all('Row')
        for row in rows:
            if ('<ElementName>us-gaap_Revenues</ElementName>' in str(row) or
                'us-gaap_SalesRevenueNet' in str(row)
                ):
                rev = xml_re(str(row))
            elif 'us-gaap_GrossProfit' in str(row):
                gross = xml_re(str(row))        
            elif ('us-gaap_CostOfRevenue' in str(row) and gross == '---' or
                'us-gaap_CostOfGoodsAndServicesSold' in str(row) and gross == '---'
                ):
                cost = xml_re(str(row))                
            elif 'us-gaap_ResearchAndDevelopmentExpense' in str(row):
                research = xml_re(str(row))           
            elif 'us-gaap_OperatingIncomeLoss' in str(row):
                oi = xml_re(str(row))      
            elif '>us-gaap_NetIncomeLoss<' in str(row) and net == '---':
                net = xml_re(str(row))                
            elif 'us-gaap_EarningsPerShareDiluted' in str(row) and eps == '---':
                eps = xml_re(str(row))
            elif 'us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding' in str(row):
                shares = xml_re(str(row))
                share_sum += shares  

        # Calculate gross if not listed
        if gross == '---':
            try:
                gross = rev - cost
            except:
                gross = rev

        # Calculate shares if not listed
        if share_sum == 0 and eps != '---':
            share_sum = round(net / eps)

        # Calculate EPS if not listed
        if eps == '---':
            eps = net / share_sum

        return rev, gross, research, oi, net, eps, share_sum, div


    # Pull balance sheet data
    def bs_html(soup):
        ''' Parse .htm balance sheet statement

        Args:
            soup (soup data of sub filings)
        Returns:
            cash (int), assets minus goodwill (int), long-term debt outstanding (int), total liabilities (int), shareholder's equity (int)
        '''
        
        # Initial values
        cash = assets = liabilities = '---'
        intangible_assets = goodwill = debt = 0

        # Loop through rows, search for row of interest
        for row in soup.table.find_all('tr'):
            tds = row.find_all('td')
            if 'this, \'defref_us-gaap_CashAndCashEquivalentsAtCarryingValue\', window' in str(tds):
                cash = html_re(str(tds))
            elif 'defref_us-gaap_Goodwill' in str(tds):
                goodwill = html_re(str(tds))
            elif ('this, \'defref_us-gaap_FiniteLivedIntangibleAssetsNet\', window' in str(tds) or
                'this, \'defref_us-gaap_IntangibleAssetsNetExcludingGoodwill\', window' in str(tds)
                ):
                intangible_assets = html_re(str(tds))
            elif ('Total assets' in str(tds) or
                'Total Assets' in str(tds)
                ):
                assets = html_re(str(tds))
            elif 'onclick="top.Show.showAR( this, \'defref_us-gaap_Liabilities\', window )' in str(tds):
                liabilities = html_re(str(tds))
            elif ('this, \'defref_us-gaap_LongTermDebtNoncurrent\', window' in str(tds) or
                'this, \'defref_us-gaap_LongTermDebt\', window' in str(tds) or
                'this, \'defref_us-gaap_LongTermDebtAndCapitalLeaseObligations\', window' in str(tds)
                ):
                debt = html_re(str(tds))
                if debt == '---':
                    debt = 0
            elif 'this, \'defref_us-gaap_LiabilitiesAndStockholdersEquity\', window' in str(tds):
                tot_liabilities = html_re(str(tds))
            elif ('this, \'defref_us-gaap_StockholdersEquity\', window' in str(tds) or
                'this, \'defref_us-gaap_StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest\', window' in str(tds)
                ):
                equity = html_re(str(tds))           

        # Net out goodwill and intangible assets from assets
        if assets != '---':
            assets -= (goodwill + intangible_assets)
        
        # Calculate liabilites from shareholder equity if not found
        if liabilities == '---':
            try:
                liabilities = tot_liabilities - equity
            except:
                pass
        
        return cash, assets, debt, liabilities, equity

    # Pull balance sheet data
    def bs_xml(soup):
        ''' Parse .xml balance sheet statement

        Args:
            soup (soup data of sub filings)
        Returns:
            cash (int), assets minus goodwill (int), long-term debt outstanding (int), total liabilities (int), shareholder's equity (int)
        '''   

        # Initial values
        cash = assets = liabilities = '---'
        intang_assets = goodwill = debt = 0

        # Loop through rows, search for row of interest
        rows = soup.find_all('Row')
        for row in rows:
            if ('us-gaap_CashCashEquivalentsAndShortTermInvestments' in str(row) or
                'us-gaap_CashAndCashEquivalentsAtCarryingValue' in str(row)
                ):
                cash = xml_re(str(row))
            elif 'us-gaap_Goodwill' in str(row):
                goodwill = xml_re(str(row))               
            elif r'us-gaap_IntangibleAssetsNetExcludingGoodwill' in str(row):
                intang_assets = xml_re(str(row))
            elif r'<ElementName>us-gaap_Assets</ElementName>' in str(row):
                assets = xml_re(str(row))      
            elif r'us-gaap_LongTermDebtNoncurrent' in str(row):
                debt = xml_re(str(row))
            elif r'<ElementName>us-gaap_Liabilities</ElementName>' in str(row):
                liabilities = xml_re(str(row))
            elif r'us-gaap_LiabilitiesAndStockholdersEquity' in str(row) and liabilities == '---':
                tot_liabilities = xml_re(str(row))
            elif r'<ElementName>us-gaap_StockholdersEquity</ElementName>' in str(row):
                equity = xml_re(str(row))                                                  

        # Net out goodwill from assets
        if assets != '---' and goodwill != '---':
            assets -= (goodwill + intang_assets)

        # Calculate liabilites from shareholder equity if not found
        if liabilities == '---':
            try:
                liabilities = tot_liabilities - equity
            except:
                pass
        
        return cash, assets, debt, liabilities, equity
        

    # Pull cash flow data
    def cf_html(soup):
        ''' Parse .htm cash flow statement

        Args:
            soup (soup data of sub filings)
        Returns:
            free cash flow (int), debt repayments (int), amount spent on share buybacks (int), amount spent on dividends (int), share based compensation (int)
        '''

        # Initial values
        share_issue = buyback = divpaid = sbc = 0
        cfo = capex = fcf = '---'
        debt_pay = []

        # Loop through rows, search for row of interest
        for row in soup.table.find_all('tr'):
            tds = row.find_all('td')
            if ('Net cash from operations' in str(tds) or
                'this, \'defref_us-gaap_NetCashProvidedByUsedInOperatingActivities\', window' in str(tds) or
                'this, \'defref_us-gaap_NetCashProvidedByUsedInOperatingActivitiesContinuingOperations\', window' in str(tds)
                ):       
                cfo = html_re(str(tds))
            elif ('Additions to property and equipment' in str(tds) or
                'this, \'defref_us-gaap_PaymentsToAcquireProductiveAssets\', window' in str(tds) or
                'this, \'defref_us-gaap_PaymentsToAcquirePropertyPlantAndEquipment\', window' in str(tds) or
                'this, \'defref_us-gaap_PaymentsForProceedsFromProductiveAssets\', window' in str(tds)
                ):
                capex = html_re(str(tds))
            elif ('this, \'defref_us-gaap_RepaymentsOfDebtMaturingInMoreThanThreeMonths\', window' in str(tds) or
                'this, \'defref_us-gaap_RepaymentsOfLongTermDebt\', window' in str(tds) or
                'this, \'defref_us-gaap_RepaymentsOfDebtAndCapitalLeaseObligations\', window' in str(tds) or
                'this, \'defref_us-gaap_InterestPaid\', window' in str(tds) or
                'this, \'defref_us-gaap_RepaymentsOfDebt\', window' in str(tds) or
                'RepaymentsOfShortTermAndLongTermBorrowings\', window' in str(tds) or
                'this, \'defref_us-gaap_RepaymentsOfLongTermDebtAndCapitalSecurities\', window' in str(tds) or
                'this, \'defref_us-gaap_InterestPaidNet\', window' in str(tds)
                ):
                debt_pay.append(html_re(str(tds)))
            elif ('Common stock repurchased' in str(tds) or
                'this, \'defref_us-gaap_PaymentsForRepurchaseOfCommonStock\', window' in str(tds)
                ):
                buyback += html_re(str(tds))
            elif ('this, \'defref_us-gaap_ProceedsFromStockOptionsExercised\', window' in str(tds) or
                'this, \'defref_us-gaap_ProceedsFromIssuanceOfCommonStock\', window' in str(tds)
                ):
                share_issue += html_re(str(tds))
            elif ('Common stock cash dividends paid' in str(tds) or
                'this, \'defref_us-gaap_PaymentsOfDividends\', window' in str(tds) or
                'PaymentsOfDividendsAndDividendEquivalentsOnCommonStockAndRestrictedStockUnits\', window' in str(tds) or
                'this, \'defref_us-gaap_PaymentsOfDividendsCommonStock\', window' in str(tds)
                ):
                divpaid = html_re(str(tds))
            elif ('this, \'defref_us-gaap_ShareBasedCompensation\', window' in str(tds) or
                'this, \'defref_us-gaap_AllocatedShareBasedCompensationExpense\', window' in str(tds)
                ):
                sbc = html_re(str(tds))

        # Calculate Free Cash Flow
        fcf = cfo - capex

        # Net out share issuance
        buyback -= share_issue
            
        # Sum debt payments, get rid of duplicates
        debt_pay = sum(set(debt_pay))

        return fcf, debt_pay, buyback, divpaid, sbc

    def cf_xml(soup):
        ''' Parse .htm cash flow statement

        Args:
            soup (soup data of sub filings)
        Returns:
            free cash flow (int), debt repayments (int), amount spent on share buybacks (int), amount spent on dividends (int), share based compensation (int)
        '''

        # Initial values
        debt_pay = share_issue = buyback = divpaid = sbc = 0
        cfo = capex = fcf = '---'

        # Loop through rows, search for row of interest
        rows = soup.find_all('Row')
        for row in rows:
            if '<ElementName>us-gaap_NetCashProvidedByUsedInOperatingActivities</ElementName>' in str(row):
                cfo = xml_re(str(row))
            elif ('us-gaap_PaymentsToAcquirePropertyPlantAndEquipment' in str(row) or
                'us-gaap_PaymentsToAcquireProductiveAssets' in str(row)
                ):
                capex = xml_re(str(row))
            elif ('us-gaap_ProceedsFromRepaymentsOfShortTermDebtMaturingInThreeMonthsOrLess' in str(row) or
                'RepaymentsOfShortTermAndLongTermBorrowings' in str(row) or
                'us-gaap_RepaymentsOfLongTermDebtAndCapitalSecurities' in str(row) or
                'us-gaap_RepaymentsOfShortTermDebt' in str(row) or
                'us-gaap_InterestPaid' in str(row) or
                'us-gaap_RepaymentsOfLongTermDebt' in str(row)
                ):
                debt_pay += xml_re(str(row))                             
            elif 'us-gaap_PaymentsForRepurchaseOfCommonStock' in str(row):
                buyback += xml_re(str(row))     
            elif ('us-gaap_ProceedsFromStockOptionsExercised' in str(row)or
                'us-gaap_ProceedsFromIssuanceOfCommonStock' in str(row)
                ):
                share_issue = xml_re(str(row))     
            elif ('us-gaap_PaymentsOfDividendsCommonStock' in str(row) or
                'us-gaap_PaymentsOfDividends' in str(row)
                ):
                divpaid = xml_re(str(row))            
            elif 'us-gaap_ShareBasedCompensation' in str(row):
                sbc = xml_re(str(row)) 
        
        # Calculate Free Cash Flow
        fcf = cfo - capex

        # Net out share issuance
        buyback -= share_issue

        return fcf, debt_pay, buyback, divpaid, sbc

    # Pull div data
    def div_html(soup):
        ''' Parse various .htm docs for div data

        Args:
            soup (soup data of sub filings)
        Returns:
            dividend (float)
        '''
        # Initial value
        div = '---'

        # If company has seperate div table
        if 'EQUITY' not in soup.find('th').text.upper():
            for row in soup.table.find_all('tr'):
                tds = row.find_all('td')
                
                # Find where quarterly data ends
                if 'onclick="top.Show.showAR( this, \'defref_us-gaap_DividendsPayableDateDeclaredDayMonthAndYear\', window )' in str(tds):
                    for index, row in enumerate(tds):
                        if not re.findall(r'\d', str(row)):
                            break
                elif 'onclick="top.Show.showAR( this, \'defref_us-gaap_CommonStockDividendsPerShareDeclared\', window )' in str(tds):
                    div = html_re(str(tds[index]))
                    return div

        # If company is a jerk and burries it (like apple)
        else:
            # If contained in equity increase/decrease statement
            if 'defref_us-gaap_IncreaseDecreaseInStockholdersEquityRollForward' in str(soup):
                for row in soup.table.find_all('tr'):
                    tds = row.find_all('td')
                    if 'this, \'defref_us-gaap_CommonStockDividendsPerShareDeclared\', window' in str(tds):
                        div = float(re.findall(r'\d+\.\d+', str(tds))[0])
                return div
            
            # If no 12 month data, for divs broken out by quarter
            elif '12 Months Ended' not in str(soup) and 'ABSTRACT' not in soup.find('th').text.upper():
                for row in soup.table.find_all('tr'):
                    tds = row.find_all('td')
                    if 'this, \'defref_us-gaap_CommonStockDividendsPerShareDeclared\', window' in str(tds):
                        div = re.findall(r'\d+\.\d+', str(tds))[0:4]
                        div = sum(list(map(float, div)))
                        return div

            # If data is not quarterly
            elif 'QUARTERLY' not in soup.text.upper() and div == '---':
                for row in soup.table.find_all('tr'):
                    tds = row.find_all('td')
                    if ('this, \'defref_us-gaap_CommonStockDividendsPerShareDeclared\', window' in str(tds) or
                        'this, \'defref_us-gaap_CommonStockDividendsPerShareCashPaid\', window' in str(tds)
                        ):
                        div = html_re(str(tds))
                        return div
            
            else:
                # Find row where 12 month data starts and generate multiplier
                index = 1
                multiplier = 1
                for header in soup.table.find_all('th'):
                    if '9 Months Ended' in str(header):
                        multiplier = 3
                        break

                    elif '12 Months Ended' in str(header):
                        multiplier = 4
                        break

                    if 'Months Ended' in str(header) and 'colspan' in str(header):
                        index_str = re.findall(r'\d', str(header))
                        index += int(index_str[0])

                # Find div data
                for row in soup.table.find_all('tr'):
                    tds = row.find_all('td')
                    
                    if ('this, \'defref_v_Cashdividendsdeclaredandpaidquarterlyperasconvertedshare\', window' in str(tds) or
                        'this, \'defref_us-gaap_CommonStockDividendsPerShareDeclared\', window' in str(tds) and 'quarterly' in str(tds)):
                        div = multiplier * html_re(str(tds[index]))
                        if multiplier == 3:
                            div += float(''.join(re.findall(r'\d+\.\d+', tds[1].text.strip())))
                
            # If data source is in one big line
            if div == '---':
                
                # Create tables with Pandas
                tables = soup.find_all('table')[0] 
                panda = pd.read_html(str(tables))
                
                # Look for table with div data
                for table in panda:
                    if ('Dividends Per Share' in table.values or
                        'DividendsPer Share' in table.values or
                        'Cash dividends per share'in table.values
                        ):                    
                        
                        # Find row with div data and pull
                        row = table.loc[table[0] == 'Cash dividends per share']
                        div = float(re.findall(r'\d+\.\d\d', str(row))[0])
                        break

        return div


    # Pull div data
    def div_xml(soup):
        ''' Parse various .htm docs for div data

        Args:
            soup (soup data of sub filings)
        Returns:
            dividend (float)
        '''
        # Initial value
        div = '---'

        # Find index for 12 month data
        cols = soup.find_all('Column')
        for index, col in enumerate(cols):
            if '12 Months Ended' in str(col):
                break
        search_id = '<Id>' + str(index + 1) + r'</Id>'

        # Go to cell with 12 month data and pull div
        rows = soup.find_all('Row')
        for row in rows:
            if '>us-gaap_CommonStockDividendsPerShareDeclared' in str(row):
                cells = row.find_all('Cell')
                for cell in cells:
                    if search_id in str(cell):
                        div = xml_re(str(cell))
                        break

        # Check if reporting quarterly div
        if 'QUARTERLY' in str(row).upper():
            div *= 4

        return div


    # Pull share data if multiple classes of shares
    def shares_html(soup, period_end):
        tables = soup.find_all('table')[0] 
        panda = pd.read_html(str(tables), match=period_end.strftime('%Y'))
        pd.set_option("display.max_rows", None, "display.max_columns", None)

        for table in panda:
            if 'Number of shares used in per share computation' in table.values:
                row = table.loc[table[0] == 'Number of shares used in per share computation']
                shares_set = set(re.findall(r'\d+', str(row.values[1])))
                shares = sum(set(map(int, shares_set)))

        return shares


    # Define statements to Parse
    intro_list = ['DOCUMENT AND ENTITY INFORMATION', 'COVER PAGE', 'COVER', 'DOCUMENT AND ENTITY INFORMATION DOCUMENT']
    income_list = ['CONSOLIDATED STATEMENTS OF EARNINGS', 'STATEMENT OF INCOME ALTERNATIVE', 'CONSOLIDATED STATEMENT OF INCOME', 'INCOME STATEMENTS', 'STATEMENT OF INCOME',
                   'CONSOLIDATED STATEMENTS OF OPERATIONS', 'STATEMENTS OF CONSOLIDATED INCOME', 'CONSOLIDATED STATEMENTS OF INCOME', 'CONSOLIDATED STATEMENT OF OPERATIONS']
    bs_list = ['BALANCE SHEETS', 'CONSOLIDATED BALANCE SHEETS', 'STATEMENT OF FINANCIAL POSITION CLASSIFIED', 'CONSOLIDATED BALANCE SHEET']
    cf_list = ['CASH FLOWS STATEMENTS', 'CONSOLIDATED STATEMENTS OF CASH FLOWS', 'STATEMENT OF CASH FLOWS INDIRECT', 'CONSOLIDATED STATEMENT OF CASH FLOWS',
               'STATEMENTS OF CONSOLIDATED CASH FLOWS']
    div_list = ['DIVIDENDS DECLARED (DETAIL)', 'CONSOLIDATED STATEMENTS OF SHAREHOLDERS\' EQUITY', 'CONSOLIDATED STATEMENTS OF SHAREHOLDERS\' EQUITY CONSOLIDATED STATEMENTS OF SHAREHOLDERS\' EQUITY (PARENTHETICAL)',
                'SHAREHOLDERS\' EQUITY', 'SHAREHOLDERS\' EQUITY AND SHARE-BASED COMPENSATION - ADDITIONAL INFORMATION (DETAIL) (USD $)', 'SHAREHOLDERS\' EQUITY - ADDITIONAL INFORMATION (DETAIL)',
                'SHAREHOLDERS\' EQUITY AND SHARE-BASED COMPENSATION - ADDITIONAL INFORMATION (DETAIL)', 'CONSOLIDATED STATEMENTS OF CHANGES IN EQUITY (PARENTHETICAL)',
                'CONSOLIDATED STATEMENTS OF CHANGES IN EQUITY CONSOLIDATED STATEMENTS OF CHANGES IN EQUITY (PARENTHETICAL)', 'STOCKHOLDERS\' EQUITY']
    shares_list = ['NET INCOME PER SHARE', 'CONSOLIDATED BALANCE SHEETS (PARENTHETICAL)']

    # Lists for data frame
    Fiscal_Period = []
    Period_End = []
    Revenue = []
    Gross_Profit = []
    Research = []
    Operating_Income = []
    Net_Profit = []
    Earnings_Per_Share = []
    Shares_Outstanding = []
    Cash = []
    Total_Assets = []
    Total_Debt = []
    Total_Liabilities = []
    SH_Equity = []
    Free_Cash_Flow = []
    Debt_Repayment = []
    Share_Buybacks = []
    Dividend_Payments = []
    Share_Based_Comp = []
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
        shares = '---'

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
                    rev, gross, research, oi, net, eps, shares, div = rev_html(soup)
                except:
                    rev_url = base_url + report.xmlfilename.text
                    content = requests.get(rev_url, headers=headers).content
                    soup = BeautifulSoup(content, 'xml')
                    rev, gross, research, oi, net, eps, shares, div = rev_xml(soup)

            # Balance sheet
            if report.shortname.text.upper() in bs_list:
                # Get URL and contents
                try:
                    bs_url = base_url + report.htmlfilename.text
                    content = requests.get(bs_url, headers=headers).content
                    soup = BeautifulSoup(content, 'html.parser')
                    cash, assets, debt, liabilities, equity = bs_html(soup)
                except:
                    bs_url = base_url + report.xmlfilename.text
                    content = requests.get(bs_url, headers=headers).content
                    soup = BeautifulSoup(content, 'xml')
                    cash, assets, debt, liabilities, equity = bs_xml(soup)

            # Cash flow
            if report.shortname.text.upper() in cf_list:
                # Get URL and contents
                try:
                    cf_url = base_url + report.htmlfilename.text
                    content = requests.get(cf_url, headers=headers).content
                    soup = BeautifulSoup(content, 'html.parser')
                    fcf, debt_pay, buyback, divpaid, sbc = cf_html(soup)
                except:
                    cf_url = base_url + report.xmlfilename.text
                    content = requests.get(cf_url, headers=headers).content
                    soup = BeautifulSoup(content, 'xml')
                    fcf, debt_pay, buyback, divpaid, sbc = cf_xml(soup)
            
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

            # Share data if not in Income Statement
            if report.shortname.text.upper() in shares_list and shares == '---':
                try:
                    intro_url = base_url + report.htmlfilename.text
                    content = requests.get(intro_url, headers=headers).content
                    soup = BeautifulSoup(content, 'html.parser')
                    shares = shares_html(soup, period_end)
                except:
                    pass

        # Add parsed data to lists for data frame
        Fiscal_Period.append(fy)
        Period_End.append(period_end)
        Revenue.append(rev)
        Gross_Profit.append(gross)
        Research.append(research)
        Operating_Income.append(oi)
        Net_Profit.append(net)
        Earnings_Per_Share.append(eps)
        Shares_Outstanding.append(shares)     
        Cash.append(cash)
        Total_Assets.append(assets)
        Total_Debt.append(debt)
        Total_Liabilities.append(liabilities)
        SH_Equity.append(equity)
        Free_Cash_Flow.append(fcf)
        Debt_Repayment.append(debt_pay)
        Share_Buybacks.append(buyback)
        Dividend_Payments.append(divpaid)
        Share_Based_Comp.append(sbc)
        Dividends.append(div)

        everything = {'FY': Fiscal_Period, 'Per': Period_End, 'Rev': Revenue, 'Gross': Gross_Profit, 'R&D': Research, 'OI': Operating_Income, 'Net': Net_Profit, 'EPS': Earnings_Per_Share,
                      'Shares': Shares_Outstanding, 'Cash': Cash, 'Assets': Total_Assets, 'Debt': Total_Debt, 'Liabilities': Total_Liabilities, 'SH_Equity': SH_Equity, 'FCF': Free_Cash_Flow,
                      'Debt_Repayment': Debt_Repayment, 'Buybacks': Share_Buybacks, 'Div_Paid': Dividend_Payments, 'SBC': Share_Based_Comp, 'Div': Dividends}
        print(everything)
        print('-'*100)
    
    # Determine if share split occured, and adjust per share metrics
    split_factor = 1
    adjusted_shares = [Shares_Outstanding[0]]
    print('-'*100)
    
    # Loop through share count and look for major differences
    for i in range(1, len(Shares_Outstanding)):
        share_ratio = (Shares_Outstanding[i-1] / Shares_Outstanding[i])
        
        # If difference found, update split factor
        if share_ratio >= 1.45 or share_ratio <= 0.7:
            split_factor *= math.ceil(share_ratio)
        
        # If there have been splits, adjust per share metrics
        if split_factor != 1:
            Earnings_Per_Share[i] = round(Earnings_Per_Share[i] / split_factor, 2)
            if Dividends[i] != '---':
                Dividends[i] = round(Dividends[i] / split_factor, 3)

        # Append shares to adjusted shares
        adjusted_shares.append(Shares_Outstanding[i] * split_factor)

    Shares_Outstanding = adjusted_shares
    
    everything = {'FY': Fiscal_Period, 'Per': Period_End, 'Rev': Revenue, 'Gross': Gross_Profit, 'R&D': Research, 'OI': Operating_Income, 'Net': Net_Profit, 'EPS': Earnings_Per_Share,
                    'Shares': Shares_Outstanding, 'Cash': Cash, 'Assets': Total_Assets, 'Debt': Total_Debt, 'Liabilities': Total_Liabilities, 'SH_Equity': SH_Equity, 'FCF': Free_Cash_Flow,
                    'Debt_Repayment': Debt_Repayment, 'Buybacks': Share_Buybacks, 'Div_Paid': Dividend_Payments, 'SBC': Share_Based_Comp, 'Div': Dividends}    
    print(everything)










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
corr() to find highest correlated with share price
'''