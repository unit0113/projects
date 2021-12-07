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
from dateutil.relativedelta import relativedelta


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

        # Get list of fillings
        self.annual_filings = annualData(self.cik, headers) 
        #quarterly_filings = quarterlyData(self, headers)   

        # Check if empty, if not, do more calcs
        if self.annual_filings != []:
            # Pull data from filings
            self.annual_data = parse_filings(self.annual_filings, 'annual', headers, self.yahoo_stock_class.splits)

            # Fill in missed divs
            self.div_edit = False
            if ('---' in self.annual_data['Div'] or
                len(self.annual_data['Div']) != len(set(self.annual_data['Div']))
                ):
                for index, (dividend, div_paid) in enumerate(zip(self.annual_data['Div'], self.annual_data['Div_Paid'])):
                    if (div_paid != 0 and dividend == '---' or
                        index + 1 != len(self.annual_data['Div']) and self.annual_data['Div_Paid'][index + 1] != 0 and self.annual_data['Div'][index] == self.annual_data['Div'][index + 1] and (self.annual_data['Div_Paid'][index] / (self.annual_data['Shares'][index] / 1000)) / (self.annual_data['Div_Paid'][index + 1] / (self.annual_data['Shares'][index + 1] / 1000)) > 1.05
                        ):
                        self.div_edit = True
                        if index == 0:
                            self.annual_data['Div'][index] = yf_div_catch(self.symbol, self.annual_data['Per'][index])
                        else:
                            self.annual_data['Div'][index] = yf_div_catch(self.symbol, self.annual_data['Per'][index], self.annual_data['Per'][index - 1], self.annual_data['Div'][index - 1])
                        print(self.annual_data['Div'])
                        print('-' * 100)

            # Update div calcs if new div found
            if self.div_edit == True:
                self.annual_data['Earnings Payout Ratio'] = divider(self.annual_data['Div'], self.annual_data['EPS'])
                self.annual_data['FCF Payout Ratio'] = divider(self.annual_data['Div'], self.annual_data['Free Cash Flow Per Share'])
                self.annual_data['FFO Payout Ratio'] = divider(self.annual_data['Div'], self.annual_data['FFO Per Share'])
                print(self.annual_data)


            # Calculate growth rates
            self.growth_rates = {}
            self.growth_rates['Revenue Growth'] = growth_rate_calc(self.annual_data['Rev'])
            self.growth_rates['Revenue per Share Growth'] = growth_rate_calc(self.annual_data['Revenue Per Share'])
            self.growth_rates['Net Income Growth'] = growth_rate_calc(self.annual_data['Net'])
            self.growth_rates['EPS Growth'] = growth_rate_calc(self.annual_data['EPS'])
            self.growth_rates['Shares Growth'] = growth_rate_calc(self.annual_data['Shares'])
            self.growth_rates['FFO Growth'] = growth_rate_calc(self.annual_data['FFO'])
            self.growth_rates['FFO per Share Growth'] = growth_rate_calc(self.annual_data['FFO Per Share'])
            self.growth_rates['FCF Growth'] = growth_rate_calc(self.annual_data['FCF'])
            self.growth_rates['FCF per Share Growth'] = growth_rate_calc(self.annual_data['Free Cash Flow Per Share'])
            self.growth_rates['Dividend Growth'] = growth_rate_calc(self.annual_data['Div'])
            self.growth_rates['Earnings Payout Ratio Growth'] = growth_rate_calc(self.annual_data['Earnings Payout Ratio'])
            self.growth_rates['FCF Payout Ratio Growth'] = growth_rate_calc(self.annual_data['FCF Payout Ratio'])
            self.growth_rates['FFO Payout Ratio Growth'] = growth_rate_calc(self.annual_data['FFO Payout Ratio'])
            self.growth_rates['FDebt Growth'] = growth_rate_calc(self.annual_data['Debt'])

            # Calculate YoY growth
            self.growth_rates['YoY Revenue Growth'] = per_over_per_growth_rate_calc(self.annual_data['Rev'])
            self.growth_rates['YoY Revenue per Share Growth'] = per_over_per_growth_rate_calc(self.annual_data['Revenue Per Share'])
            self.growth_rates['YoY Net Income Growth'] = per_over_per_growth_rate_calc(self.annual_data['Net'])
            self.growth_rates['YoY EPS Growth'] = per_over_per_growth_rate_calc(self.annual_data['EPS'])
            self.growth_rates['YoY Shares Growth'] = per_over_per_growth_rate_calc(self.annual_data['Shares'])
            self.growth_rates['YoY FFO Growth'] = per_over_per_growth_rate_calc(self.annual_data['FFO'])
            self.growth_rates['YoY FFO per Share Growth'] = per_over_per_growth_rate_calc(self.annual_data['FFO Per Share'])
            self.growth_rates['YoY FCF Growth'] = per_over_per_growth_rate_calc(self.annual_data['FCF'])
            self.growth_rates['YoY FCF per Share Growth'] = per_over_per_growth_rate_calc(self.annual_data['Free Cash Flow Per Share'])
            self.growth_rates['YoY Dividend Growth'] = per_over_per_growth_rate_calc(self.annual_data['Div'])

            print('-' * 100)
            print(self.growth_rates)

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
    if pd.Timestamp(per[-1]) > split_list_date[0]:
        return False
    else:
        split_return = [1]
        split_factor = 1
        split_index = 0
        
        # Check if splits more recent than last annual report
        for split_date in split_list_date:
            if split_date > pd.Timestamp(per[0]):
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
            if split_list_date[split_index] < pd.Timestamp(per[-1]):
                end_list = [split_factor] * ((len(per) - i))
                split_return.extend(end_list)
                break

            if split_list_date[split_index] < pd.Timestamp(per[i-1]) and split_list_date[split_index] > pd.Timestamp(per[i]):
                split_factor *= split_list_splits[split_index]
                split_index += 1
            split_return.append(split_factor)      

        # Check if latest split is on report day
        if split_list_date[0] == pd.Timestamp(per[0]) + pd.Timedelta(days=1):
            split_return[0] = 1.0

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


def yf_div_catch(ticker, per, pre_per=None, pre_div=None):
    ''' Pulls div data from YF if div not reported

    Args:
        Ticker (str)
        Period end date (str)
    Returns:
        Dividend (float)
    '''  

    # Pull div data
    stock = yf.Ticker(ticker)

    # Intitiate period end and period start as datetime object
    per_edit = per.replace('.', '')
    date_per = datetime. strptime(per_edit, '%b %d, %Y')
    date_per_start = date_per - relativedelta(years=1)
    offset = 0

    # Reverse divi dataframs
    rev_divs = stock.dividends.iloc[::-1]
    divs = rev_divs.loc[date_per:date_per_start]

    # Determine if offset required
    offset_needed = False
    if pre_per != None and pre_div != None:
        pre_per_edit = pre_per.replace('.', '')
        pre_date_per = datetime. strptime(pre_per_edit, '%b %d, %Y')
        pre_date_per_start = pre_date_per - relativedelta(years=1)
        dividends_pre = list(rev_divs[pre_date_per:pre_date_per_start])
        dividends_pre = round(sum(dividends_pre), 3)
        if dividends_pre != pre_div:
            offset_needed = True

    # Adjust dates if offset needed
    if offset_needed == True:
        for offset in range(1, 4):
            pre_date_per_new = pre_date_per + relativedelta(months=offset)
            pre_date_per_start_new = pre_date_per_start + relativedelta(months=offset)
            divs_new = rev_divs.loc[pre_date_per_new:pre_date_per_start_new]
            divs_new = round(sum(list(divs_new)), 3)
            if divs_new == pre_div:
                break

    # Pull div data accounting for possible offset
    dividends = list(rev_divs[date_per + relativedelta(months=offset):date_per_start + relativedelta(months=offset)])
    div = round(sum(dividends), 3)
    print('-' * 100)
    print(f'Dividend of {div} caught by YF')
    return div


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


def divider(num, denom):
    ''' Divide two lists elementwise accounting for empty data

    Args:
        Numerator (List)
        Denominator (List)
    Returns:
        Result (List)
    '''

    # Convert empty results to 0
    adj_num = [0.0 if type(elem) == str else float(elem) for elem in num]
    adj_denom = [0.0 if type(elem) == str else float(elem) for elem in denom]
    
    # return empty results if all of num is '---'
    if sum(adj_num) == 0:
        return ['---'] * len(num)

    # Calculate return and convert 0's back to empty result
    with np.errstate(invalid='ignore', divide='ignore'):
        result = np.divide(adj_num, adj_denom, out=np.zeros_like(adj_num), where=adj_denom!=0.0)
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
    adj_first = [0.0 if type(elem) == str else elem for elem in first]
    adj_second = [0.0 if type(elem) == str else elem for elem in second]

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
    adj_first = [0.0 if type(elem) == str else elem for elem in first]
    adj_second = [0.0 if type(elem) == str else elem for elem in second]

    # Calculate return and convert 0's back to empty result
    result = np.add(adj_first, adj_second)
    result = ['---' if elem == 0 else round(elem, 2) for elem in result]

    return result


def parse_filings(filings, type, headers, splits):    
    ''' Parse filings and create datafram from the data

    Args:
        filings (lst of dicts contains filings data), type (str, options of annual or quarterly), headers (dict, user agent for accessing SEC website w/o getting yelled at)
    Return:
        results (dataframe)       
    '''

    # Define statements to Parse
    intro_list = ['DOCUMENT AND ENTITY INFORMATION', 'COVER PAGE', 'COVER', 'DOCUMENT AND ENTITY INFORMATION DOCUMENT', 'COVER PAGE COVER PAGE', 'DEI DOCUMENT', 'COVER DOCUMENT', 'DOCUMENT INFORMATION STATEMENT',
                  'DOCUMENT ENTITY INFORMATION', 'DOCUMENT AND ENTITY INFORMATION DOCUMENT AND ENTITY INFORMATION', 'COVER COVER', 'DOCUMENT', 'DOCUMENT ENTITY INFORMATION DOCUMENT',
                  'DOCUMENT AND ENTITY INFORMATION (PARENTHETICALS)', 'ENTITY INFORMATION']
    income_list = ['CONSOLIDATED STATEMENTS OF EARNINGS', 'STATEMENT OF INCOME ALTERNATIVE', 'CONSOLIDATED STATEMENT OF INCOME', 'INCOME STATEMENTS', 'STATEMENT OF INCOME', 'CONDENSED CONSOLIDATED INCOME STATEMENT',
                   'CONSOLIDATED STATEMENTS OF OPERATIONS', 'STATEMENTS OF CONSOLIDATED INCOME', 'CONSOLIDATED STATEMENTS OF INCOME', 'CONSOLIDATED STATEMENT OF OPERATIONS', 
                   'CONSOLIDATED STATEMENTS OF EARNINGS (LOSSES)', 'CONSOLIDATED INCOME STATEMENTS', 'CONSOLIDATED STATEMENTS OF OPERATIONS CONSOLIDATED STATEMENTS OF OPERATIONS',
                   'CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS', 'CONSOLIDATED STATEMENTS OF NET INCOME', 'CONSOLIDATED AND COMBINED STATEMENTS OF OPERATIONS', 'CONSOLIDATED STATEMENT OF EARNINGS',
                   'CONSOLIDATED STATEMENTS OF OPERATIONS AND COMPREHENSIVE INCOME (LOSS)', 'CONSOLIDATED STATEMENTS OF OPERATIONS AND COMPREHENSIVE INCOME', 'CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS AND COMPREHENSIVE INCOME (LOSS)',
                   'CONSOLIDATED STATEMENTS OF OPERATIONS AND COMPREHENSIVE LOSS', 'CONSOLIDATED STATEMENTS OF OPERATIONS AND OTHER COMPREHENSIVE LOSS', 'STATEMENTS OF OPERATIONS', 'STATEMENTS OF CONSOLIDATED EARNINGS',
                   'CONSOLIDATED RESULTS OF OPERATIONS', 'CONDENSED CONSOLIDATED STATEMENTS OF EARNINGS', 'STATEMENT OF CONSOLIDATED INCOME', 'CONSOLIDATED STATEMENTS OF INCOME AND COMPREHENSIVE INCOME', 'CONSOLIDATED STATEMENTS OF INCOME (LOSS)',
                   'CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME', 'CONSOLIDATED STATEMENTS OF EARNINGS (LOSS)', 'CONSOLDIATED STATEMENTS OF OPERATIONS', 'CONSOLIDATED INCOME STATEMENT',
                   'CONDENSED CONSOLIDATED STATEMENTS OF INCOME', 'STATEMENTS OF CONSOLIDATED OPERATIONS', 'STATEMENTS OF INCOME']
    bs_list = ['BALANCE SHEETS', 'CONSOLIDATED BALANCE SHEETS', 'STATEMENT OF FINANCIAL POSITION CLASSIFIED', 'CONSOLIDATED BALANCE SHEET', 'CONDENSED CONSOLIDATED BALANCE SHEETS',
               'CONSOLIDATED AND COMBINED BALANCE SHEETS', 'CONSOLIDATED STATEMENTS OF FINANCIAL POSITION', 'BALANCE SHEET', 'CONSOLIDATED FINANCIAL POSITION', 'CONSOLIDATED STATEMENTS OF FINANCIAL CONDITION',
               'CONSOLIDATED STATEMENT OF FINANCIAL POSITION', 'CONDENSED CONSOLIDATED BALANCE SHEET', 'CONDENSED CONSOLIDATED STATEMENTS OF FINANCIAL CONDITION']
    cf_list = ['CASH FLOWS STATEMENTS', 'CONSOLIDATED STATEMENTS OF CASH FLOWS', 'STATEMENT OF CASH FLOWS INDIRECT', 'CONSOLIDATED STATEMENT OF CASH FLOWS', 'STATEMENTS OF CASH FLOWS',
               'STATEMENTS OF CONSOLIDATED CASH FLOWS', 'CONSOLIDATED CASH FLOWS STATEMENTS', 'CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS', 'CONSOLIDATED AND COMBINED STATEMENTS OF CASH FLOWS', 'CONSOLIDATED STATEMENT OF CASH FLOW',
               'STATEMENT OF CASH FLOWS', 'CONSOLIDATED STATEMENTS OF CASH FLOW', 'CONSOLIDATED  STATEMENTS OF CASH FLOWS', 'STATEMENT OF CONSOLIDATED CASH FLOWS', 'CONSOLIDATED STATEMENTS OF CASH FLOWS (UNAUDITED)',
               'CONSOLDIATED STATEMENTS OF CASH FLOWS', 'CONSOLIDATED CASH FLOW STATEMENT', 'CONDENSED CONSOLIDATED CASH FLOW STATEMENT', 'CONSOLIDATED CASH FLOW STATEMENTS']
    div_list = ['DIVIDENDS DECLARED (DETAIL)', 'CONSOLIDATED STATEMENTS OF SHAREHOLDERS\' EQUITY', 'CONSOLIDATED STATEMENTS OF SHAREHOLDERS\' EQUITY CONSOLIDATED STATEMENTS OF SHAREHOLDERS\' EQUITY (PARENTHETICAL)',
                'SHAREHOLDERS\' EQUITY', 'SHAREHOLDERS\' EQUITY AND SHARE-BASED COMPENSATION - ADDITIONAL INFORMATION (DETAIL) (USD $)', 'SHAREHOLDERS\' EQUITY - ADDITIONAL INFORMATION (DETAIL)',
                'SHAREHOLDERS\' EQUITY AND SHARE-BASED COMPENSATION - ADDITIONAL INFORMATION (DETAIL)', 'CONSOLIDATED STATEMENTS OF CHANGES IN EQUITY (PARENTHETICAL)', 'STATEMENT OF EQUITY (PARENTHETICAL)',
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
                'STATEMENTS OF CONSOLIDATED SHAREHOLDERS\' EQUITY AND COMPREHENSIVE INCOME (PARENTHETICAL)', 'SUMMARY BY QUARTER (SCHEDULE OF UNAUDITED OPERATING RESULTS BY QUARTER) (DETAILS)', 'SHAREHOLDERS\' EQUITY (DETAILS)',
                'CONSOLIDATED STATEMENTS OF CHANGES IN STOCKHOLDERS\' EQUITY (PARENTHETICAL)', 'CONSOLIDATED STATEMENT OF CHANGES IN EQUITY', 'QUARTERLY FINANCIAL INFORMATION', 'CONSOLIDATED STATEMENTS OF SHAREHOLDERS\' EQUITY (PARENTHETICAL)',
                'CONSOLIDATED STATEMENT OF SHAREHOLDERS\' EQUITY (PARENTHETICAL)', 'CONSOLIDATED STATEMENT OF SHAREHOLDERS\' EQUITY CONSOLIDATED STATEMENT OF SHAREHOLDERS\'S EQUITY (PARENTHETICAL)',
                'CONSOLIDATED STATEMENT OF SHAREHOLDERS\' INVESTMENT (PARENTHETICAL)', 'SUPPLEMENTARY FINANCIAL INFORMATION (UNAUDITED) (DETAILS)', 'STATEMENTS OF STOCKHOLDERS\' EQUITY (PARENTHETICAL)',
                'STATEMENTS OF STOCKHOLDERS\' EQUITY STATEMENTS OF STOCKHOLDERS\' EQUITY (PARENTHETICAL)', 'STATEMENT OF CONSOLIDATED STOCKHOLDERS\'S EQUITY (PARENTHETICAL)', 'CONDENSED CONSOLIDATED STATEMENTS OF CHANGES IN EQUITY',
                'CONDENSED CONSOLIDATED STATEMENTS OF CHANGES IN EQUITY (PARENTHETICAL)', 'CONSOLIDATED STATEMENTS OF CHANGES IN EQUITY', 'CONSOLIDATED STATEMENTS OF SHAREHOLDERS??? EQUITY (PARENTHETICAL)',
                'CONSOLIDATED STATEMENTS OF SHAREHOLDERS??? EQUITY PARENTHETICAL', 'QUARTERLY RESULTS OF OPERATIONS (UNAUDITED) (DETAILS)', 'QUARTERLY FINANCIAL INFORMATION (DETAIL)', 'QUARTERLY RESULTS OF OPERATIONS (SCHEDULE OF QUARTERLY RESULTS OF OPERATIONS) (DETAILS)',
                'CONSOLIDATED STATEMENTS OF SHAREHOLDERS EQUITY (PARENTHETICAL)', 'CONSOLIDATED STATEMENTS OF STOCKHOLDERS\' EQUITY CONSOLIDATED STATEMENTS OF STOCKHOLDERS\' EQUITY (PARENTHETICAL)',
                'STOCK-BASED COMPENSATION - STOCK OPTION ASSUMPTIONS (DETAILS)', 'STOCK-BASED COMPENSATION (STOCK OPTION ASSUMPTIONS) (DETAILS)', 'CHANGES IN CONSOLIDATED SHAREHOLDERS\' EQUITY', 'STOCKHOLDERS\' EQUITY (COMMON STOCK DIVIDENDS) (DETAILS)',
                'STOCKHOLDERS\' EQUITY (DEFICIT) (NARRATIVE) (DETAILS)', 'STOCKHOLDERS\' EQUITY - DIVIDENDS (DETAILS)', 'STOCKHOLDERS\' EQUITY (DETAIL 2)', 'STOCKHOLDERS\' EQUITY (DETAILS 2)', 'STOCKHOLDERS\' EQUITY (DETAILS)',
                'SHAREHOLDERS\' EQUITY (NARRATIVE) (DETAILS)', 'EQUITY AND ACCUMULATED OTHER COMPREHENSIVE INCOME (LOSS), NET - SCHEDULE OF DIVIDENDS (DETAILS)', 'EQUITY AND ACCUMULATED OTHER COMPREHENSIVE LOSS, NET (SCHEDULE OF DIVIDENDS) (DETAILS)',
                'EQUITY AND ACCUMULATED OTHER COMPREHENSIVE LOSS, NET (SCHEDULE OF DIVIDENDS DECLARED AND PAYABLE) (DETAILS)', 'CONSOLIDATED STATEMENT OF CHANGES IN EQUITY CONSOLIDATED STATEMENT OF CHANGES IN EQUITY (PARENTHETICAL)',
                'CONSOLIDATED STATEMENT OF CHANGES IN EQUITY (PARENTHETICAL)', 'STOCKHOLDERS\' EQUITY (EARNINGS PER SHARE DATA) (DETAILS)', 'CONSOLIDATED STATEMENTS OF EQUITY (PARENTHETICALS)', 'COMMON AND PREFERRED SHARES COMMON SHARES (DETAILS)',
                'CONSOLIDATED STATEMENTS OF CHANGES IN COMMON STOCKHOLDERS\' INVESTMENT (PARENTHETICAL)', 'CONSOLIDATED STATEMENTS OF CHANGES IN SHAREHOLDERS EQUITY (PARENTHETICAL)', 'CONSOLIDATED STATEMENTS OF CHANGES IN SHAREHOLDERS EQUITY AND COMPREHENSIVE INCOME (PARENTHETICAL)',
                'CONSOLIDATED STATEMENT OF CHANGES IN EQUITY (PARANTHETICAL)', 'CONSOLIDATED STATEMENT OF CHANGES IN STOCKHOLDERS\' EQUITY (PARANTHETICAL)', 'INCOME TAXES - FEDERAL INCOME TAX TREATMENT OF COMMON DIVIDENDS (DETAILS)',
                'DIVIDENDS (DETAILS)', 'DIVIDENDS', 'SHAREHOLDERS\' EQUITY - (NARRATIVE) (DETAILS)', 'STOCKHOLDERS\' EQUITY (DETAILS TEXTUAL)', 'SHAREHOLDERS\' EQUITY (DETAILS 3)', 'SHAREHOLDERS\' EQUITY (TABLES)',
                'CONSOLIDATED STATEMENTS OF CHANGES IN SHAREHOLDERS\' INVESTMENT (PARENTHETICAL)', 'CONSOLDIATED STATEMENTS OF CHANGES IN SHAREHOLDERS\' INVESTMENT (PARENTHETICAL)', 'CONSOLIDATED STATEMENTS OF CHANGES IN SHAREHOLDERS\' INVESTMENT  (PARENTHETICAL)',
                'STOCKHOLDERS\' EQUITY MATTERS - DIVIDENDS DECLARED (DETAILS)', 'CONSOLIDATED STATEMENT OF STOCKHOLDERS\' EQUITY (PARENTHETICALS)', 'UNAUDITED QUARTERLY DATA (DETAILS)', 'CONSOLIDATED STATEMENT OF CHANGES IN SHAREHOLDERS\' EQUITY (PARENTHETICAL)',
                'CONSOLIDATED STATEMENTS OF CHANGES IN SHAREOWNERS\' EQUITY (PARENTHETICAL)', 'CONSOLIDATED STATEMENTS OF  EQUITY (PARENTHETICAL)', 'SELECTED QUARTERLY DATA (UNAUDITED) (DETAILS)', 'SELECTED QUARTERLY DATA (DETAILS)',
                'CAPITAL STOCK - DIVIDENDS PAID (DETAIL)', 'DIVIDENDS PAID (DETAIL)', 'CAPITAL STOCK (SCHEDULE OF DIVIDENDS PAID BY COMPANY) (DETAILS)', 'CAPITAL STOCK (NARRATIVE) (DETAILS)', 'STATEMENTS OF EQUITY (PARENTHETICAL)',
                'SELECTED QUARTERLY DATA (SCHEDULED OF QUARTERLY FINANCIAL INFORMATION) (DETAILS)', 'TOTAL EQUITY - DIVIDENDS (DETAILS)', 'TOTAL EQUITY - COMMON STOCK DIVIDENDS PER SHARE (DETAILS)',
                'TOTAL EQUITY (DIVIDENDS AND SHARE REPURCHASES) (DETAILS)', 'QUARTERLY RESULTS (DETAILS)', 'QUARTERLY RESULTS (UNAUDITED) - (QUARTERLY RESULTS) (DETAILS)', 'SHAREHOLDERS\' EQUITY - DIVIDENDS DECLARED (DETAILS)',
                'DISTRIBUTIONS PAID AND PAYABLE - DISTRIBUTIONS TO COMMON STOCKHOLDERS (DETAILS)', 'DISTRIBUTIONS PAID AND PAYABLE - COMMON STOCK (DETAILS)', 'DISTRIBUTIONS PAID AND PAYABLE (DETAILS)', 'DISTRIBUTIONS PAID AND PAYABLE (DETAIL)',
                'STOCKHOLDERS EQUITY (PER SHARE DISTRIBUTIONS) (DETAIL)', 'DIVIDENDS (PER SHARE DISTRIBUTIONS) (DETAIL)', 'DIVIDENDS (DETAILS 1)', 'DIVIDENDS (DETAIL)', 'EARNINGS PER COMMON SHARE ATTRIBUTABLE TO PFIZER INC. COMMON SHAREHOLDERS (TABLES)',
                'CONSOLIDATED STATEMENTS OF STOCKHOLDERS\' (DEFICIT) EQUITY (PARENTHETICAL)', 'CONSOLIDATED STATEMENTS OF STOCKHOLDERS (DEFICIT) EQUITY (PARENTHETICAL)', 'QUARTERLY FINANCIAL DATA (TABLES)',
                'EARNINGS PER SHARE AND UNIT AND SHAREHOLDERS\' EQUITY AND CAPITAL - EARNINGS PER SHARE AND UNIT (DETAILS)', 'EARNINGS PER SHARE AND UNIT AND SHAREHOLDERS\' EQUITY AND CAPITAL (DETAILS)', 'CAPITAL STOCK DIVIDENDS (DETAILS)',
                'CONSOLIDATED STATEMENT OF SHAREOWNERS\' EQUITY (PARENTHETICAL)', 'SHAREHOLDERS\' EQUITY (SCHEDULE OF DIVIDENDS PAID AND DIVIDENDS DECLARED) (DETAILS)', 'CONSOLIDATED STATEMENTS OF EQUITY',
                'EQUITY - DIVIDENDS (DETAILS)', 'CONSOLIDATED STATEMENTS OF EQUITY (USD $) (PARENTHETICAL)', 'QUARTERLY FINANCIAL INFORMATION (UNAUDITED) (DETAILS)']
    eps_catch_list = ['EARNINGS PER SHARE', 'EARNINGS (LOSS) PER SHARE', 'STOCKHOLDERS\' EQUITY', 'EARNINGS PER SHARE (DETAILS)', 'EARNINGS PER SHARE (DETAIL)', 'EARNING PER SHARE (DETAIL)', 'EARNINGS PER SHARE (BASIC AND DILUTED WEIGHTED AVERAGE SHARES OUTSTANDING) (DETAILS)',
                      'EARNINGS PER SHARE (SCHEDULE OF COMPUTATION OF BASIC AND DILUTED EARNINGS PER SHARE) (DETAIL)']
    share_catch_list = ['CONSOLIDATED BALANCE SHEETS (PARENTHETICAL)', 'CONSOLIDATED BALANCE SHEET (PARENTHETICAL)', 'CONSOLIDATED BALANCE SHEETS (PARANTHETICAL)', 'CONSOLIDATED BALANCE SHEET CONSOLIDATED BALANCE SHEET (PARENTHETICAL)',
                        'CONSOLIDATED BALANCE SHEET (PARENTHETICALS)', 'CONDENSED CONSOLIDATED BALANCE SHEET (PARENTHETICAL)', 'CONSOLIDATED BALANCE SHEETS (PARENTHETICAL)', 'CONSOLIDATED BALANCE SHEETS - PARENTHETICAL INFO', 'CONSOLIDATED BALANCE SHEETS',
                        'EQUITY (COMMON STOCK AND CLASS B STOCK CHANGES IN NUMBER OF SHARES ISSUED, HELD IN TREASURY AND OUTSTANDING) (DETAILS)', 'EQUITY (COMMON STOCK CHANGES IN NUMBER OF SHARES ISSUED, HELD IN TREASURY AND OUTSTANDING) (DETAILS)']

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
    Current_Assets = []
    Total_Assets = []
    Total_Debt = []
    Current_Liabilities = []
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
        fy = period_end = name = rev = gross = research = oi = net = eps = shares = div = ffo = cash = cur_assets = assets = debt = cur_liabilities = liabilities = equity = fcf = buyback = divpaid = sbc = '---'
        div_prob = 0
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
                    if name_diff < 50 and name != '---':
                        diff_comp_flag = True
                        print(f'Name comparision between {name} and {names[-1]} failed with a ratio of {name_diff}')
                break

        # Loop through each report with the 'myreports' a second time because some companies put the document summary at the end
        for report in reports.find_all('report')[:-1]:
            # Break if name comp failed
            if diff_comp_flag == True:
                break

            # Income Statement
            if report.shortname.text.upper() in income_list and rev == '---':
                # Create URL and call parser function
                try:
                    rev_url = base_url + report.htmlfilename.text
                    rev, gross, research, oi, net, eps, shares_return, div, ffo = sp.rev_htm(rev_url, headers, period_end)
                    if shares_return != '---' and shares_return != 0:
                        shares = shares_return
                except:
                    rev_url = base_url + report.xmlfilename.text
                    rev, gross, research, oi, net, eps, shares_return, div, ffo = sp.rev_xml(rev_url, headers)
                    if shares_return != '---' and shares_return != 0:
                        shares = shares_return

            # Balance sheet
            elif report.shortname.text.upper() in bs_list and cash == '---':
                # Create URL and call parser function
                try:
                    bs_url = base_url + report.htmlfilename.text
                    cash, cur_assets, assets, debt, cur_liabilities, liabilities, equity = sp.bs_htm(bs_url, headers, period_end)
                except:
                    bs_url = base_url + report.xmlfilename.text
                    cash, cur_assets, assets, debt, cur_liabilities, liabilities, equity = sp.bs_xml(bs_url, headers)

            # Cash flow
            elif report.shortname.text.upper() in cf_list and fcf == '---':
                # Create URL and call parser function
                try:
                    cf_url = base_url + report.htmlfilename.text
                    fcf, buyback, divpaid, sbc = sp.cf_htm(cf_url, headers, period_end)
                except:
                    cf_url = base_url + report.xmlfilename.text
                    fcf, buyback, divpaid, sbc = sp.cf_xml(cf_url, headers)
            
            # Dividends
            elif report.shortname.text.upper() in div_list and div == '---':
                # Create URL and call parser function
                try:
                    div_url = base_url + report.htmlfilename.text
                    div_result = sp.div_htm(div_url, headers, period_end)
                    if div_result != '---' and len(Dividends) > 1 and Dividends[-1] != '---' and 'EQUITY' in report.shortname.text.upper() and (div_result == Dividends[-1] or div_result / Dividends[-1] < 0.8):
                        div_prob = max(div_result, div_prob)
                    else:
                        div = div_result
                except:
                    div_url = base_url + report.xmlfilename.text
                    div = sp.div_xml(div_url, headers)

            # EPS/div catcher
            elif report.shortname.text.upper() in eps_catch_list and (div == '---' and divpaid != 0 or eps == '---' or shares == '---'):
                # Create URL and call parser function
                    try:
                        catch_url = base_url + report.htmlfilename.text
                        eps_result, div_result, share_result = sp.eps_catch_htm(catch_url, headers, period_end)
                    except:
                        catch_url = base_url + report.xmlfilename.text
                        eps_result, div_result, share_result = sp.eps_catch_xml(catch_url, headers)
                    # Update EPS or div if result found/needed
                    if eps == '---' and eps_result != '---':
                        eps = eps_result
                    if div == '---' and div_result != '---':
                        div = div_result
                    if shares == '---' and share_result != '---':
                            shares = share_result
                
            # Shares if not reported on income statement
            if report.shortname.text.upper() in share_catch_list and shares == '---' or report.shortname.text.upper() in share_catch_list and shares == 0:
                # Create URL and call parser function
                try:
                    catch_url = base_url + report.htmlfilename.text
                    shares = sp.share_catch_htm(catch_url, headers, period_end)
                except:
                    catch_url = base_url + report.xmlfilename.text
                    shares = sp.share_catch_xml(catch_url, headers, period_end)
            
            # Break if all data found
            if (rev != '---' and cash != '---' and fcf != '---' and shares != '---' and eps != '---' and div != '---' and divpaid > 0 or
                rev != '---' and cash != '---' and fcf != '---' and shares != '---' and eps != '---' and divpaid == 0
                ):
                break
                
        # Check for errors in FY pull (company's fault)
        if len(Fiscal_Period) > 0:
            if int(Fiscal_Period[-1]) - int(fy) > 1 and str(int(Period_End[-1][-4:]) - 1) in period_end:
                fy = str(int(Period_End[-1][-4:]) - 1)
                print('FY error caught')

        # Check for repeat data
        if len(Fiscal_Period) != 0:
            if fy == Fiscal_Period[-1] or fy == '---':
                continue

        # Check for name/company difference
        if diff_comp_flag == True:
            break

        # Set div equal to found div when it's the same as previous and no better value found
        if div == '---' and div_prob != 0:
            div = div_prob

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
        Current_Assets.append(cur_assets)
        Total_Assets.append(assets)
        Total_Debt.append(debt)
        Current_Liabilities.append(cur_liabilities)
        Total_Liabilities.append(liabilities)
        SH_Equity.append(equity)
        Free_Cash_Flow.append(fcf)
        Share_Buybacks.append(buyback)
        Dividend_Payments.append(divpaid)
        Share_Based_Comp.append(sbc)
        Dividends.append(div)
        names.append(name)

        everything = {'FY': Fiscal_Period, 'Per': Period_End, 'Rev': Revenue, 'Gross': Gross_Profit, 'R&D': Research, 'OI': Operating_Income, 'Net': Net_Profit, 'EPS': Earnings_Per_Share,
                      'Shares': Shares_Outstanding, 'FFO': Funds_From_Operations, 'Cash': Cash, 'Current Assets': Current_Assets, 'Assets': Total_Assets, 'Debt': Total_Debt, 'Current Liabilities': Current_Liabilities, 'Liabilities': Total_Liabilities, 'SH_Equity': SH_Equity, 'FCF': Free_Cash_Flow,
                      'Buybacks': Share_Buybacks, 'Div_Paid': Dividend_Payments, 'SBC': Share_Based_Comp, 'Div': Dividends}
        print(everything)
        print('-'*100)
    
    # Adjust share and per share data for stock splits
    split_list = split_factor_calc(splits, Period_End)
    if split_list is not False:
        Earnings_Per_Share = split_adjuster(split_list, Earnings_Per_Share)
        Dividends = split_adjuster(split_list, Dividends)
        Shares_Outstanding = split_adjuster(split_list, Shares_Outstanding, shares=True)    

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
    Return_on_Captial_Employed = divider(Net_Profit, subtractor(Total_Assets, Current_Liabilities))

    # Book Value
    Book_Value = subtractor(Total_Assets, Total_Liabilities)
    Book_Value_Per_Share = divider(Book_Value, divider(Shares_Outstanding, [1000] * len(Shares_Outstanding)))

    # Other
    Debt_to_Profit_Ratio = divider(Total_Debt, Net_Profit)
    Debt_to_Equity_Ratio = divider(Total_Debt, SH_Equity)
    Current_Ratio = divider(Current_Assets, Current_Liabilities)


    everything = {'FY': Fiscal_Period, 'Per': Period_End, 'Rev': Revenue, 'Gross': Gross_Profit, 'R&D': Research, 'OI': Operating_Income, 'Net': Net_Profit, 'EPS': Earnings_Per_Share,
                    'Shares': Shares_Outstanding, 'FFO': Funds_From_Operations, 'Cash': Cash, 'Current Assets': Current_Assets, 'Assets': Total_Assets, 'Debt': Total_Debt, 'Current Liabilities': Current_Liabilities, 'Liabilities': Total_Liabilities, 'SH_Equity': SH_Equity, 'FCF': Free_Cash_Flow,
                    'Buybacks': Share_Buybacks, 'Div_Paid': Dividend_Payments, 'SBC': Share_Based_Comp, 'Div': Dividends, 'Revenue Per Share': Revenue_Per_Share, 'FFO Per Share': FFO_Per_Share,
                    'Free Cash Flow Per Share': FCF_Per_Share, 'Earnings Payout Ratio': Earning_Payout_Ratio, 'FCF Payout Ratio': FCF_Payout_Ratio, 'FFO Payout Ratio': FFO_Payout_Ratio, 'Gross Margin': Gross_Margin,
                    'Operating Margin': Operating_Margin, 'Net Margin': Net_Margin, 'FFO Margin': FFO_Margin, 'FCF Margin': FCF_Margin, 'SBC Margin': SBC_Margin, 'R&D Margin': Research_Margin, 'ROA': Return_on_Assets,
                    'ROE': Return_on_Equity, 'ROIC': Return_on_Invested_Capital, 'ROCE': Return_on_Captial_Employed, 'Book Value': Book_Value, 'Book Value Per Share': Book_Value_Per_Share, 'Debt to Profit Ratio': Debt_to_Profit_Ratio,
                    'Debt to Equity Ratio': Debt_to_Equity_Ratio,'Current Ratio': Current_Ratio}    
    print(everything)

    return everything


def growth_rate_calc(numbers):
    ''' Calculates 1, 3, 5, and 10 year growth rates

    Args:
        List of yearly numbers
    Returns:
        List of 1, 3, 5, and 10 year growth rates
    '''

    # Replace empty values with 0
    adj_numbers = [0 if type(elem) == str else elem for elem in numbers]

    # Initial values
    one_year = three_year = five_year = ten_year = '---'

    # One year growth
    if len(numbers) > 1 and adj_numbers[0] > 0 and adj_numbers[1] > 0:
        one_year = round((adj_numbers[0] - adj_numbers[1]) / adj_numbers[1], 4)
    # Three year growth
    if len(adj_numbers) > 3 and adj_numbers[0] > 0 and adj_numbers[3] > 0:
        three_year = round(((adj_numbers[0] / adj_numbers[3]) ** (1 / 3)) - 1, 4)
    # Five year growth
    if len(adj_numbers) > 5 and adj_numbers[0] > 0 and adj_numbers[5] > 0:
        five_year = round(((adj_numbers[0] / adj_numbers[5]) ** (1 / 5)) - 1, 4)
    # Ten year growth
    if len(adj_numbers) > 10 and adj_numbers[0] > 0 and adj_numbers[10] > 0:
        ten_year = round(((adj_numbers[0] / adj_numbers[10]) ** (1 / 10)) - 1, 4)

    return one_year, three_year, five_year, ten_year


def per_over_per_growth_rate_calc(numbers):
    ''' Calculates period over period growth

    Args:
        List of numbers
    Returns:
        List of growth rates
    '''

    # Return nothing if length is only 1
    if len(numbers) == 1:
        return '---'

    adj_numbers = [0 if type(elem) == str else elem for elem in numbers]

    # Calc growth
    results = []    
    for i in range(1, len(adj_numbers)):
        if adj_numbers[i] != 0:
            growth = (adj_numbers[i-1] - adj_numbers[i]) / adj_numbers[i]
            results.append(round(growth, 4))
        else:
            results.append(0)

    adj_results = ['---' if elem == 0 else elem for elem in results]

    return adj_results


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
find a way to parse for companies that don't report an xml summary (MAIN)
determine avg price per FY for use in price to whatever calcs
calculate relative price to indexes chart data
percent increase/decrease from starting point for comparision (shares, eps, fcf, etc...)
'''