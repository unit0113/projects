import statement_parser as sp
import requests
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
from bs4 import BeautifulSoup
import pandas as pd
import time
import sys
import re
import math
import yfinance as yf
import numpy as np
from fuzzywuzzy import fuzz


class StockData:
    """ Class to store stock data for requested tickers, takes in initial input from edgargui
    """    
    
    def __init__(self, symbol, cik):
        """ Takes in ticker and CIK, then initiates the class

        Args:
            Stocker ticker (str)
            CIK number (int)
        """ 

        self.symbol = symbol.upper()
        self.cik = cik

        # Get yahoo data, and initiate split factor list
        self.yahoo_stock_class = yahoo_pull(self.symbol)
        #self.split_list = split_factor_calc(self.yahoo_stock_class.splits, self.annual_data['Per'])

        # Get list of fillings
        self.annual_filings = annualData(self.cik, headers) 
        #quarterly_filings = quarterlyData(self, headers)   

        # Pull data from filings
        self.annual_data = parse_filings(self.annual_filings, 'annual', headers, self.yahoo_stock_class.splits)


def network_check_decorator(num, max_attempts=5):
    ''' Retries failed network connections

    Args:
        Failure identifier (int)
        *optional* max number of reattempts, default = 5
    Returns:
        N/A
    '''
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


def yahoo_pull(ticker):
    ''' Pull yahoo finance data

    Args:
        Ticker (str)
    Returns:
        Stock info (dict)
    '''
    # pull data using yfinance
    stock = yf.Ticker(ticker)

    return stock


def split_factor_calc(splits, per):
    ''' Calculate a list of stock split factors

    Args:
        Dataframe of dates and splits (datafram)
        List of report end dates (datetime object)

    Returns:
        split factors by year (list)
    '''

    # Check if there are any splits, if none, return False
    if splits.empty:
        return False

    # Convert strings to datetime objects
    per = [datetime.strptime(period.replace('.', ''), '%b %d, %Y').date() for period in per]

    # Pull and adjust split data from yf
    split_list_date = splits.index.values.tolist()
    split_list_date = list(pd.to_datetime(split_list_date))
    split_list_date.reverse()
    split_list_splits = list(splits)
    split_list_splits.reverse()

    # Check if there are any relevant splits
    if per[-1] > split_list_date[0]:
        return False
    else:
        split_return = [1]
        split_factor = 1
        split_index = 0
        
        # Check if splits more recent than last annual report
        for split_date in split_list_date:
            if split_date > per[0]:
                split_factor *= split_list_splits[split_index]
                split_return[0] *= split_factor
                split_index += 1
            else:
                break
        
        # Create list of split factors
        for i in range(1, len(per)):
            # If end of splits
            if split_index > len(split_list_date) - 1:
                end_list = [split_factor] * ((len(per) - i))
                split_return.extend(end_list)
                break 

            # If end of relevant splits
            if split_list_date[split_index] < per[-1]:
                end_list = [split_factor] * ((len(per) - i))
                split_return.extend(end_list)
                break

            if split_list_date[split_index] < per[i-1] and split_list_date[split_index] > per[i]:
                split_factor *= split_list_splits[split_index]
                split_index += 1
            split_return.append(split_factor)      


        return split_return


def split_adjuster(split_factor, values, shares=False):
    ''' Adjust data pulls for stock splits

    Args:
        Split factors (list)
        Values to adjust (list)
        If values are shares (bool), optional, default is False
    Returns:
        Adjusted values (list)
    '''    

    # Adjust values in case of empty
    values = [0 if elem == '---' else elem for elem in values]

    # Adjust values based on split factor
    if shares == False:
        adj_values = np.divide(values, split_factor)
        adj_values = list(np.around(np.array(adj_values),3))
    else:
        adj_values = np.multiply(values, split_factor)
        adj_values = list(map(int, adj_values))

    # Unadjust div values in case of empty
    adj_values = ['---' if elem == 0 else elem for elem in adj_values]

    return adj_values


def getFilings(cik, period, limit, headers):
    ''' Get list of filings from Edgar

    Args:
        CIK number (int)
        Type of fillings (str), ex. '10-k', '10-q'
        How many filings to grab (int)
        User agent for SEC
    Returns:
        Filings (list of dicts)
    '''

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
    ''' Get quarterly data from Edgar

    Args:
        CIK number (int)
        User agent for SEC
    Returns:
        Quarterly filings (list of dicts)
    '''

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
    ''' Get annual data from Edgar

    Args:
        CIK number (int)
        User agent for SEC
    Returns:
        Annual filings (list of dicts)
    '''

    # Get filings
    try:
        filings = getFilings(cik, '10-k', 11, headers)
    except:
        filings = getFilings(cik, '20-f', 11, headers)    
    
    return filings


def parse_filings(filings, type, headers, splits):    
    ''' Parse filings and create datafram from the data

    Args:
        filings (lst of dicts contains filings data), type (str, options of annual or quarterly), headers (dict, user agent for accessing SEC website w/o getting yelled at)
    Return:
        results (dataframe)       
    '''
    
    # Define statements to Parse
    intro_list = ['DOCUMENT AND ENTITY INFORMATION', 'COVER PAGE', 'COVER', 'DOCUMENT AND ENTITY INFORMATION DOCUMENT', 'COVER PAGE COVER PAGE', 'DEI DOCUMENT', 'COVER DOCUMENT', 'DOCUMENT INFORMATION STATEMENT',
                  'DOCUMENT ENTITY INFORMATION', 'DOCUMENT AND ENTITY INFORMATION DOCUMENT AND ENTITY INFORMATION', 'COVER COVER']
    income_list = ['CONSOLIDATED STATEMENTS OF EARNINGS', 'STATEMENT OF INCOME ALTERNATIVE', 'CONSOLIDATED STATEMENT OF INCOME', 'INCOME STATEMENTS', 'STATEMENT OF INCOME',
                   'CONSOLIDATED STATEMENTS OF OPERATIONS', 'STATEMENTS OF CONSOLIDATED INCOME', 'CONSOLIDATED STATEMENTS OF INCOME', 'CONSOLIDATED STATEMENT OF OPERATIONS', 
                   'CONSOLIDATED STATEMENTS OF EARNINGS (LOSSES)', 'CONSOLIDATED INCOME STATEMENTS', 'CONSOLIDATED STATEMENTS OF OPERATIONS CONSOLIDATED STATEMENTS OF OPERATIONS',
                   'CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS', 'CONSOLIDATED STATEMENTS OF NET INCOME', 'CONSOLIDATED AND COMBINED STATEMENTS OF OPERATIONS', 'CONSOLIDATED STATEMENT OF EARNINGS',
                   'CONSOLIDATED STATEMENTS OF OPERATIONS AND COMPREHENSIVE INCOME (LOSS)', 'CONSOLIDATED STATEMENTS OF OPERATIONS AND COMPREHENSIVE INCOME', 'CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS AND COMPREHENSIVE INCOME (LOSS)',
                   'CONSOLIDATED STATEMENTS OF OPERATIONS AND COMPREHENSIVE LOSS', 'CONSOLIDATED STATEMENTS OF OPERATIONS AND OTHER COMPREHENSIVE LOSS', 'STATEMENTS OF OPERATIONS', 'STATEMENTS OF CONSOLIDATED EARNINGS']
    bs_list = ['BALANCE SHEETS', 'CONSOLIDATED BALANCE SHEETS', 'STATEMENT OF FINANCIAL POSITION CLASSIFIED', 'CONSOLIDATED BALANCE SHEET', 'CONDENSED CONSOLIDATED BALANCE SHEETS',
               'CONSOLIDATED AND COMBINED BALANCE SHEETS', 'CONSOLIDATED STATEMENTS OF FINANCIAL POSITION', 'BALANCE SHEET']
    cf_list = ['CASH FLOWS STATEMENTS', 'CONSOLIDATED STATEMENTS OF CASH FLOWS', 'STATEMENT OF CASH FLOWS INDIRECT', 'CONSOLIDATED STATEMENT OF CASH FLOWS',
               'STATEMENTS OF CONSOLIDATED CASH FLOWS', 'CONSOLIDATED CASH FLOWS STATEMENTS', 'CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS', 'CONSOLIDATED AND COMBINED STATEMENTS OF CASH FLOWS', 'CONSOLIDATED STATEMENT OF CASH FLOW',
               'STATEMENT OF CASH FLOWS']
    div_list = ['DIVIDENDS DECLARED (DETAIL)', 'CONSOLIDATED STATEMENTS OF SHAREHOLDERS\' EQUITY', 'CONSOLIDATED STATEMENTS OF SHAREHOLDERS\' EQUITY CONSOLIDATED STATEMENTS OF SHAREHOLDERS\' EQUITY (PARENTHETICAL)',
                'SHAREHOLDERS\' EQUITY', 'SHAREHOLDERS\' EQUITY AND SHARE-BASED COMPENSATION - ADDITIONAL INFORMATION (DETAIL) (USD $)', 'SHAREHOLDERS\' EQUITY - ADDITIONAL INFORMATION (DETAIL)',
                'SHAREHOLDERS\' EQUITY AND SHARE-BASED COMPENSATION - ADDITIONAL INFORMATION (DETAIL)', 'CONSOLIDATED STATEMENTS OF CHANGES IN EQUITY (PARENTHETICAL)',
                'CONSOLIDATED STATEMENTS OF CHANGES IN EQUITY CONSOLIDATED STATEMENTS OF CHANGES IN EQUITY (PARENTHETICAL)', 'STOCKHOLDERS\' EQUITY', 'CONSOLIDATED STATEMENTS OF STOCKHOLDERS\' EQUITY (PARENTHETICAL)',
                'CONDENSED STATEMENTS OF STOCKHOLDERS\' EQUITY (PARENTHETICAL)', 'CONDENSED STATEMENTS OF STOCKHOLDERS\' EQUITY AND COMPREHENSIVE INCOME (PARENTHETICAL)', 'CONSOLIDATED STATEMENTS OF STOCKHOLDERS\' EQUITY AND COMPREHENSIVE INCOME (PARENTHETICAL)',
                'STOCKHOLDERS\' EQUITY - ADDITIONAL INFORMATION (DETAILS)', 'CONSOLIDATED STATEMENT OF EQUITY', 'CONSOLIDATED STATEMENTS OF STOCKHOLDERS EQUITY (PARENTHETICAL)', 'QUARTERLY FINANCIAL INFORMATION (UNAUDITED)',
                'STOCKHOLDERS\' EQUITY (DIVIDENDS AND DISTRIBUTIONS) (DETAILS)', 'STOCKHOLDERS\' EQUITY (DIVIDENDS) (DETAILS)', 'CONSOLIDATED STATEMENTS OF STOCKHOLDERS\' EQUITY (PARENTHETICALS)',
                'CONSOLIDATED STATEMENTS OF EQUITY (PARENTHETICAL)', 'EQUITY - CASH DIVIDENDS (DETAILS)', 'SHAREHOLDERS\' EQUITY DIVIDENDS (DETAILS)', 'CONSOLIDATED STATEMENT OF CHANGES IN EQUITY (PARENTHETICALS)',
                'CONSOLIDATED STATEMENT OF EQUITY (PARENTHETICAL)', 'EQUITY (CHANGES IN EQUITY) (DETAILS)', 'EQUITY (TABLES)', 'CONSOLIDATED STATEMENTS OF EQUITY / CAPITAL (PARENTHETICAL)',
                'CONSOLIDATED STATEMENTS OF EQUITY/CAPITAL (PARENTHETICAL)', 'CONSOLIDATED AND COMBINED STATEMENTS OF EQUITY (PARENTHETICAL)', 'DIVIDENDS', 'CONSOLIDATED STATEMENTS OF CHANGES IN SHAREHOLDERS\' EQUITY (PARENTHETICAL)',
                'CONSOLIDATED AND COMBINED STATEMENTS OF EQUITY AND PARTNERSHIP CAPITAL (PARENTHETICAL)', 'EQUITY', 'DISTRIBUTIONS (DETAILS)', 'DISTRIBUTIONS', 'CONSOLIDATED STATEMENTS OF COMMON SHAREHOLDERS\' EQUITY',
                'CONSOLIDATED STATEMENTS OF COMMON SHAREHOLDERS\' EQUITY (PARENTHETICAL)', 'SUPPLEMENTAL EQUITY AND COMPREHENSIVE INCOME INFORMATION - DIVIDENDS AND TRANSFER OF OWNERSHIP INTEREST (DETAILS)',
                'CONSOLIDATED STATEMENT OF SHAREOWNERS EQUITY', 'UNAUDITED QUARTERLY FINANCIAL INFORMATION (DETAILS)', 'UNAUDITED SELECTED QUARTERLY DATA (DETAILS)', 'STATEMENTS OF CONSOLIDATED SHAREHOLDERS\' EQUITY (PARENTHETICAL)',
                'STATEMENTS OF CONSOLIDATED SHAREHOLDERS\' EQUITY AND COMPREHENSIVE INCOME (PARENTHETICAL)', 'SUMMARY BY QUARTER (SCHEDULE OF UNAUDITED OPERATING RESULTS BY QUARTER) (DETAILS)',
                'CONSOLIDATED STATEMENTS OF CHANGES IN STOCKHOLDERS\' EQUITY (PARENTHETICAL)', 'CONSOLIDATED STATEMENT OF CHANGES IN EQUITY', 'QUARTERLY FINANCIAL INFORMATION', 'CONSOLIDATED STATEMENTS OF SHAREHOLDERS\' EQUITY (PARENTHETICAL)',
                'CONSOLIDATED STATEMENT OF SHAREHOLDERS\' EQUITY (PARENTHETICAL)', 'CONSOLIDATED STATEMENT OF SHAREHOLDERS\' EQUITY CONSOLIDATED STATEMENT OF SHAREHOLDERS\'S EQUITY (PARENTHETICAL)',
                'CONSOLIDATED STATEMENT OF SHAREHOLDERS\' INVESTMENT (PARENTHETICAL)', 'SUPPLEMENTARY FINANCIAL INFORMATION (UNAUDITED) (DETAILS)', 'STATEMENTS OF STOCKHOLDERS\' EQUITY (PARENTHETICAL)',
                'STATEMENTS OF STOCKHOLDERS\' EQUITY STATEMENTS OF STOCKHOLDERS\' EQUITY (PARENTHETICAL)', 'STATEMENT OF CONSOLIDATED STOCKHOLDERS\'S EQUITY (PARENTHETICAL)', 'CONDENSED CONSOLIDATED STATEMENTS OF CHANGES IN EQUITY',
                'CONDENSED CONSOLIDATED STATEMENTS OF CHANGES IN EQUITY (PARENTHETICAL)', 'CONSOLIDATED STATEMENTS OF CHANGES IN EQUITY', 'CONSOLIDATED STATEMENTS OF SHAREHOLDERS??? EQUITY (PARENTHETICAL)',
                'CONSOLIDATED STATEMENTS OF SHAREHOLDERS??? EQUITY PARENTHETICAL', 'QUARTERLY RESULTS OF OPERATIONS (UNAUDITED) (DETAILS)', 'QUARTERLY FINANCIAL INFORMATION (DETAIL)', 'QUARTERLY RESULTS OF OPERATIONS (SCHEDULE OF QUARTERLY RESULTS OF OPERATIONS) (DETAILS)']
    eps_catch_list = ['EARNINGS PER SHARE', 'EARNINGS (LOSS) PER SHARE', 'STOCKHOLDERS\' EQUITY', 'EARNINGS PER SHARE (DETAILS)']
    share_catch_list = ['CONSOLIDATED BALANCE SHEETS (PARENTHETICAL)', 'CONSOLIDATED BALANCE SHEET (PARENTHETICAL)']

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
    Funds_From_Operations = []
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
    names = []
    

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

        for file in content['directory']['item']:  
            # Grab the filing summary url
            if file['name'] == 'FilingSummary.xml':
                xml_summary = r"https://www.sec.gov" + content['directory']['name'] + "/" + file['name']
                print(xml_summary)
                
                # Define a new base url
                base_url = xml_summary.replace('FilingSummary.xml', '')
                break

        reports = pull_filing_2(xml_summary)
        
        # Initial values
        fy = period_end = name = rev = gross = research = oi = net = eps = shares = div = ffo = cash = assets = debt = liabilities = equity = fcf = debt_pay = buyback = divpaid = sbc = '---'
        diff_comp_flag = False

        # Loop through each report with the 'myreports' tag but avoid the last one as this will cause an error
        for report in reports.find_all('report')[:-1]:

            # Summary table
            if report.shortname.text.upper() in intro_list and fy == '---':
                # Create URL and call parser function
                try:
                    intro_url = base_url + report.htmlfilename.text
                    fy, period_end, name = sp.sum_htm(intro_url, headers)
                except:
                    intro_url = base_url + report.xmlfilename.text
                    fy, period_end, name= sp.sum_xml(intro_url, headers)
        
                # Check if different company than earlier (mergers)
                if len(names) != 0:
                    name_diff = fuzz.token_set_ratio(name,names[-1])
                    if name_diff < 55 and name != '---':
                        diff_comp_flag = True
                        print(f'Name comparision between {name} and {names[-1]} failed with a ratio of {name_diff}')
                        break

            # Income Statement
            if report.shortname.text.upper() in income_list and rev == '---':
                # Create URL and call parser function
                try:
                    rev_url = base_url + report.htmlfilename.text
                    rev, gross, research, oi, net, eps, shares_return, div, ffo = sp.rev_htm(rev_url, headers, period_end)
                    if shares == '---' or shares == 0:
                        shares = shares_return
                except:
                    rev_url = base_url + report.xmlfilename.text
                    rev, gross, research, oi, net, eps, shares_return, div, ffo = sp.rev_xml(rev_url, headers)
                    if shares == '---' or shares == 0:
                        shares = shares_return

            # Balance sheet
            if report.shortname.text.upper() in bs_list and cash == '---':
                # Create URL and call parser function
                try:
                    bs_url = base_url + report.htmlfilename.text
                    cash, assets, debt, liabilities, equity = sp.bs_htm(bs_url, headers, period_end)
                except:
                    bs_url = base_url + report.xmlfilename.text
                    cash, assets, debt, liabilities, equity = sp.bs_xml(bs_url, headers)

            # Cash flow
            if report.shortname.text.upper() in cf_list and fcf == '---':
                # Create URL and call parser function
                try:
                    cf_url = base_url + report.htmlfilename.text
                    fcf, debt_pay, buyback, divpaid, sbc = sp.cf_htm(cf_url, headers, period_end)
                except:
                    cf_url = base_url + report.xmlfilename.text
                    fcf, debt_pay, buyback, divpaid, sbc = sp.cf_xml(cf_url, headers)
            
            # Dividends
            if report.shortname.text.upper() in div_list and div == '---':
                # Create URL and call parser function
                try:
                    div_url = base_url + report.htmlfilename.text
                    div = sp.div_htm(div_url, headers, period_end)
                except:
                    div_url = base_url + report.xmlfilename.text
                    div = sp.div_xml(div_url, headers)

            # EPS/div catcher
            if report.shortname.text.upper() in eps_catch_list and div == '---' and divpaid != 0 or report.shortname.text.upper() in eps_catch_list and eps == '---':
                # Create URL and call parser function
                    try:
                        catch_url = base_url + report.htmlfilename.text
                        eps_result, div_result, share_result = sp.eps_catch_htm(catch_url, headers, period_end)
                        if shares == 0 and share_result != '---':
                            shares = share_result
                    except:
                        catch_url = base_url + report.xmlfilename.text
                        eps_result, div_result = sp.eps_catch_xml(catch_url, headers, period_end)
                    # Update EPS or div if result found/needed
                    if eps == '---' and eps_result != '---':
                        eps = eps_result
                        # Calc shares if not found previously
                        if shares == 0 and net != '---':
                            shares = abs(round(net * 1000 / eps))
                    if div == '---' and div_result != '---':
                        div = div_result
                
            # Shares if not reported on income statement
            if report.shortname.text.upper() in share_catch_list and shares == '---' or report.shortname.text.upper() in share_catch_list and shares == 0:
                # Create URL and call parser function
                try:
                    catch_url = base_url + report.htmlfilename.text
                    shares = sp.share_catch_htm(catch_url, headers, period_end)
                except:
                    catch_url = base_url + report.xmlfilename.text
                    shares = sp.share_catch_xml(catch_url, headers, period_end)

        # Check for repeat data
        if len(Fiscal_Period) != 0:
            if fy == Fiscal_Period[-1] or fy == '---':
                continue

        # Check for name/company difference
        if diff_comp_flag == True:
            break

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
        Funds_From_Operations.append(ffo)     
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
        names.append(name)

        everything = {'FY': Fiscal_Period, 'Per': Period_End, 'Rev': Revenue, 'Gross': Gross_Profit, 'R&D': Research, 'OI': Operating_Income, 'Net': Net_Profit, 'EPS': Earnings_Per_Share,
                      'Shares': Shares_Outstanding, 'FFO': Funds_From_Operations, 'Cash': Cash, 'Assets': Total_Assets, 'Debt': Total_Debt, 'Liabilities': Total_Liabilities, 'SH_Equity': SH_Equity, 'FCF': Free_Cash_Flow,
                      'Debt_Repayment': Debt_Repayment, 'Buybacks': Share_Buybacks, 'Div_Paid': Dividend_Payments, 'SBC': Share_Based_Comp, 'Div': Dividends}
        print(everything)
        print('-'*100)
    
    # Adjust share and per share data for stock splits
    split_list = split_factor_calc(splits, Period_End)
    if split_list is not False:
        Earnings_Per_Share = split_adjuster(split_list, Earnings_Per_Share)
        Dividends = split_adjuster(split_list, Dividends)
        Shares_Outstanding = split_adjuster(split_list, Shares_Outstanding, shares=True)    


    def divider(num, denom):
        ''' Divide two lists elementwise accounting for empty data

        Args:
            Numerator (List)
            Denominator (List)
        Returns:
            Result (List)
        '''

        # Convert empty results to 0
        adj_num = [0 if elem == '---' else elem for elem in num]
        adj_denom = [0 if elem == '---' else elem for elem in denom]

        # return empty results if all of num is '---'
        if sum(adj_num) == 0:
            return ['---'] * len(num)

        # Calculate return and convert 0's back to empty result
        result = np.divide(adj_num, adj_denom)
        result = ['---' if elem == 0 else round(elem, 4) for elem in result]

        return result
    

    def subtractor(first, second):
        ''' Subtract two lists elementwise accounting for empty data

        Args:
            List being subtracted from (List)
            List being subtracted (List)
        Returns:
            Result (List)
        '''
        
        # Convert empty results to 0
        adj_first = [0 if elem == '---' else elem for elem in first]
        adj_second = [0 if elem == '---' else elem for elem in second]

        # Calculate return and convert 0's back to empty result
        result = np.subtract(adj_first, adj_second)
        result = ['---' if elem == 0 else round(elem, 2) for elem in result]

        return result


    def adder(first, second):
        ''' Add two lists elementwise accounting for empty data

        Args:
            List being added (List)
            Other list being added (List)
        Returns:
            Result (List)
        '''
        
        # Convert empty results to 0
        adj_first = [0 if elem == '---' else elem for elem in first]
        adj_second = [0 if elem == '---' else elem for elem in second]

        # Calculate return and convert 0's back to empty result
        result = np.add(adj_first, adj_second)
        result = ['---' if elem == 0 else round(elem, 2) for elem in result]

        return result

    # Further calculations of margins and payout ratios and what not
    # Per share data
    Revenue_Per_Share = divider(Revenue, divider(Shares_Outstanding, [1000] * len(Shares_Outstanding)))
    FFO_Per_Share = divider(Funds_From_Operations, divider(Shares_Outstanding, [1000] * len(Shares_Outstanding)))
    FCF_Per_Share = divider(Free_Cash_Flow, divider(Shares_Outstanding, [1000] * len(Shares_Outstanding)))
    
    # Payout ratios
    Earning_Payout_Ratio = divider(Dividends, Earnings_Per_Share)
    FCF_Payout_Ratio = divider(Dividends, FCF_Per_Share)
    FFO_Payout_Ratio = divider(Dividends, FFO_Per_Share)
    
    # Margins
    Gross_Margin = divider(Gross_Profit, Revenue)
    Operating_Margin = divider(Operating_Income, Revenue)
    Net_Margin = divider(Net_Profit, Revenue)
    FFO_Margin = divider(Funds_From_Operations, Revenue)
    FCF_Margin = divider(Free_Cash_Flow, Revenue)
    SBC_Margin = divider(Share_Based_Comp, Revenue)
    Research_Margin = divider(Research, Revenue)
    
    # Returns
    Return_on_Assets = divider(Net_Profit, Total_Assets)
    Return_on_Equity = divider(Net_Profit, SH_Equity)
    nopat = subtractor(Net_Profit, Dividend_Payments)
    ic = adder(Total_Debt, SH_Equity)
    Return_on_Invested_Capital = divider(nopat, ic)

    # Book Value
    Book_Value = subtractor(Total_Assets, Total_Liabilities)
    Book_Value_Per_Share = divider(Book_Value, divider(Shares_Outstanding, [1000] * len(Shares_Outstanding)))





    everything = {'FY': Fiscal_Period, 'Per': Period_End, 'Rev': Revenue, 'Gross': Gross_Profit, 'R&D': Research, 'OI': Operating_Income, 'Net': Net_Profit, 'EPS': Earnings_Per_Share,
                    'Shares': Shares_Outstanding, 'FFO': Funds_From_Operations, 'Cash': Cash, 'Assets': Total_Assets, 'Debt': Total_Debt, 'Liabilities': Total_Liabilities, 'SH_Equity': SH_Equity, 'FCF': Free_Cash_Flow,
                    'Debt_Repayment': Debt_Repayment, 'Buybacks': Share_Buybacks, 'Div_Paid': Dividend_Payments, 'SBC': Share_Based_Comp, 'Div': Dividends, 'Revenue Per Share': Revenue_Per_Share, 'FFO Per Share': FFO_Per_Share,
                    'Free Cash Flow Per Share': FCF_Per_Share, 'Earnings Payout Ratio': Earning_Payout_Ratio, 'FCF Payout Ratio': FCF_Payout_Ratio, 'FFO Payout Ratio': FFO_Payout_Ratio, 'Gross Margin': Gross_Margin,
                    'Operating Margin': Operating_Margin, 'Net Margin': Net_Margin, 'FFO Margin': FFO_Margin, 'FCF Margin': FCF_Margin, 'SBC Margin': SBC_Margin, 'R&D Margin': Research_Margin, 'ROA': Return_on_Assets,
                    'ROE': Return_on_Equity, 'ROIC': Return_on_Invested_Capital, 'Book Value': Book_Value, 'Book Value Per Share': Book_Value_Per_Share}    
    print(everything)

    return everything








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
figure out how to get SBC from table
'''