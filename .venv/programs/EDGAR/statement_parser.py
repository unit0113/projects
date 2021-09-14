import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime
import pandas as pd
import math


# RE for .htm filings
def html_re(string):
    ''' RE syntax and error checking for pulling numeric information from data cell on .htm filings

    Args:
        Target string (str)
    Return:
        Results (int/float)       
    '''

    # Pull numeric data from string
    if 'toggleNextSibling(this)' not in string:
        obj = re.search(r'(?:<td class=\"(?:num|nump)\">)(?:.*?\(?)(\d*,?\d*,?,?\d*.?\d+)(?:\)?<span>)', string, re.M)
        try:
            result = obj.group(1).replace(',', '')
        except:
            return '---'
    else:
        obj = re.search(r'(?:\">)(?:\$? ?\(?)(\d*,?\d*,?,?\d*.?\d+)(?:\)?</a>)', string, re.M)
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
    

# Check for possible negative number when not expected
def check_neg(string, value, type='htm'):
    ''' Checks for unexpected negative numbers

    Args:
        Target string (str)
        Value that we're checking (int/float)
        Type of filing ('htm' or 'xml')
    Return:
        Adjusted value (int/float)       
    '''
    # Create search string based on value, look for char before and after
    if type == 'htm':
        re_str = '.' + "{:,}".format(value) + '.'
    else:
        re_str = '.' + str(value) + '.'
    obj = re.search(re_str, string, re.M)
    
    # Check if negative
    if '(' in obj.group(0) and ')' in obj.group(0):
        return (value * -1)
    else:
        return value


# Find correct column for data
def column_finder_annual_htm(soup):
    ''' Determine which column of data contains desired data for htm pages

    Args:
        Soup (soup data of sub filings)
    Returns:
        Column number (int)        
    '''

    # Extract colmn header string
    head = soup.table.find_all('tr')[0]
    
    if '12 Months Ended' in str(soup):        
        # Find correct column if multiple things before 12 months data
        colm = re.findall(r'(?:colspan=\"(\d\d?)\")(?!>12 Months Ended)', str(head), re.M)
        return sum(map(int, colm))
    else:
        # Find correct column accounting for empties prior to data
        colm = re.search(r'(?:colspan=\"(\d\d?)\")', str(head), re.M)
        return int(colm.group(1))


def sum_htm(sum_url, headers):
    ''' Parse .htm document information

    Args:
        URL of statement
        User agent for SEC
    Returns:
        Fiscal year (int)
        Fiscal period end date (datetime object)        
    '''

    # Get data from site
    content = requests.get(sum_url, headers=headers).content
    soup = BeautifulSoup(content, 'html.parser')

    # Initial values
    fy = period_end = '---'

    # Loop through rows, search for FY and period end date
    for row in soup.table.find_all('tr'):
        tds = row.find_all('td')
        if r"this, 'defref_dei_DocumentFiscalYearFocus', window" in str(tds):
            fy = int(tds[1].text)
        if r"this, 'defref_dei_DocumentPeriodEndDate', window" in str(tds):
            period_end_str = tds[1].text.strip()
            period_end = datetime.strptime(period_end_str, '%b. %d,  %Y') 

    return fy, period_end


'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


def rev_htm(rev_url, headers):    
    ''' Parse .htm income statement

    Args:
        URL of statement
        User agent for SEC
    Returns:
        Revenue (int)
        Gross profit (int)
        R&D costs (int)
        Operating income (int)
        Net income (int)
        Earnings per share (float)
        Share count (int)
        Dividend (float)      
    '''

    # Get data from site
    content = requests.get(rev_url, headers=headers).content
    soup = BeautifulSoup(content, 'html.parser')

    # Initial values
    rev = gross = oi = net = eps = cost = shares = div = '---'
    share_sum = research = non_attributable_net = 0

    # Find which column has 12 month data
    colm = column_finder_annual_htm(soup)
 
    # Loop through rows, search for row of interest
    for row in soup.table.find_all('tr'):
        tds = row.find_all('td')
        if (r"this, 'defref_us-gaap_Revenues', window" in str(tds) or
            r"this, 'defref_us-gaap_SalesRevenueNet', window" in str(tds) and rev == '---' or
            r'defref_us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax' in str(tds) and rev == '---'
            ):
            rev = html_re(str(tds[colm]))
        elif r'defref_us-gaap_GrossProfit' in str(tds):
            gross = html_re(str(tds[colm]))
            if gross != '---':
                gross = check_neg(str(tds), gross)
        elif (r"this, 'defref_us-gaap_ResearchAndDevelopmentExpense', window" in str(tds) or
              r"this, 'defref_amzn_TechnologyAndContentExpense', window" in str(tds)
              ):
            research = html_re(str(tds[colm]))
        elif r"this, 'defref_us-gaap_OperatingIncomeLoss', window" in str(tds):
            oi = html_re(str(tds[colm]))
            if oi != '---':
                oi = check_neg(str(tds), oi)
        elif ('this, \'defref_us-gaap_NetIncomeLoss\', window' in str(tds) and net == '---' or
              'this, \'defref_us-gaap_ProfitLoss\', window' in str(tds) and net == '---'
            ):
            net = html_re(str(tds[colm]))
            if net != '---':
                net = check_neg(str(tds), net)
        elif r"this, 'defref_us-gaap_NetIncomeLossAttributableToNoncontrollingInterest', window" in str(tds):
            non_attributable_net = html_re(str(tds[colm]))
            if net != '---':
                if net > 0:
                    net -= non_attributable_net
                else:
                    net += non_attributable_net
        elif ('EarningsPerShareDiluted' in str(tds) and eps == '---' or
              'this, \'defref_us-gaap_EarningsPerShareBasicAndDiluted\', window' in str(tds) and eps == '---' or
              r"this, 'defref_us-gaap_IncomeLossFromContinuingOperationsPerBasicAndDilutedShare', window" in str(tds) and eps == '---'
              ):
            eps = html_re(str(tds[colm]))
            if eps != '---':
                eps = check_neg(str(tds), eps)
        elif (r"this, 'defref_us-gaap_CostOfRevenue', window" in str(tds) and cost == '---' or
              r"this, 'defref_us-gaap_CostOfGoodsSold', window" in str(tds) and cost == '---' or
              r"this, 'defref_us-gaap_CostOfGoodsAndServicesSold', window" in str(tds) and cost == '---'
              ):
            cost = html_re(str(tds[colm]))
        elif (r"this, 'defref_us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding', window" in str(tds) or
              r"this, 'defref_us-gaap_WeightedAverageNumberOfShareOutstandingBasicAndDiluted', window" in str(tds) or
              r"this, 'defref_tsla_WeightedAverageNumberOfSharesOutstandingBasicAndDilutedOne', window" in str(tds)
              ):
            shares = html_re(str(tds[colm]))
            share_sum += shares
        elif (r"this, 'defref_us-gaap_CommonStockDividendsPerShareDeclared', window" in str(tds) or
              r"this, 'defref_us-gaap_CommonStockDividendsPerShareCashPaid', window" in str(tds)
              ):
            div = html_re(str(tds[colm]))
            
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

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

def bs_htm(bs_url, headers):    
    ''' Parse .htm balance sheet statement

    Args:
        URL of statement
        User agent for SEC
    Returns:
        Cash (int)
        Assets minus goodwill (int)
        Long-term debt outstanding (int)
        Total liabilities (int)
        Shareholder's equity (int)
    '''
    
    # Get data from site
    content = requests.get(bs_url, headers=headers).content
    soup = BeautifulSoup(content, 'html.parser')

    # Initial values
    equity = cash = assets = liabilities = '---'
    intangible_assets = goodwill = debt = 0
    intangible_assets_set = set()

    # Find which column has 12 month data
    colm = column_finder_annual_htm(soup)

    # Loop through rows, search for row of interest
    for row in soup.table.find_all('tr'):
        tds = row.find_all('td')
        if 'this, \'defref_us-gaap_CashAndCashEquivalentsAtCarryingValue\', window' in str(tds):
            cash = html_re(str(tds[colm]))
        elif 'defref_us-gaap_Goodwill' in str(tds):
            goodwill = html_re(str(tds[colm]))
        elif ('this, \'defref_us-gaap_FiniteLivedIntangibleAssetsNet\', window' in str(tds) or
              'this, \'defref_us-gaap_IntangibleAssetsNetExcludingGoodwill\', window' in str(tds) or
              r"this, 'defref_us-gaap_IndefiniteLivedIntangibleAssetsExcludingGoodwill', window" in str(tds)
              ):
            intangible_assets = html_re(str(tds[colm]))
            if intangible_assets != '---':
                intangible_assets_set.add(intangible_assets)
        elif ('Total assets' in str(tds) or
              'Total Assets' in str(tds) or
              r"this, 'defref_us-gaap_Assets', window" in str(tds)
              ):
            assets = html_re(str(tds[colm]))
        elif 'onclick="top.Show.showAR( this, \'defref_us-gaap_Liabilities\', window )' in str(tds):
            liabilities = html_re(str(tds[colm]))
        elif ('this, \'defref_us-gaap_LongTermDebtNoncurrent\', window' in str(tds) or
              'this, \'defref_us-gaap_LongTermDebt\', window' in str(tds) or
              'this, \'defref_us-gaap_LongTermDebtAndCapitalLeaseObligations\', window' in str(tds)
              ):
            debt = html_re(str(tds[colm]))
            if debt == '---':
                debt = 0
        elif 'this, \'defref_us-gaap_LiabilitiesAndStockholdersEquity\', window' in str(tds):
            tot_liabilities = html_re(str(tds[colm]))
        elif ('this, \'defref_us-gaap_StockholdersEquity\', window' in str(tds) and equity =='---' or
              'this, \'defref_us-gaap_StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest\', window' in str(tds) and equity == '---'
              ):
            equity = html_re(str(tds[colm]))
            if equity != '---':
                equity = check_neg(str(tds), equity)          

    # Calculate liabilites from shareholder equity if not found
    if liabilities == '---' and tot_liabilities != '---' and equity != '---':
        liabilities = tot_liabilities - equity

    # Net out goodwill from assets
    if assets != '---' and goodwill != '---':
        assets -= (goodwill + sum(intangible_assets_set))
    
    return cash, assets, debt, liabilities, equity

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

def cf_htm(cf_url, headers):
    ''' Parse .htm cash flow statement

    Args:
        URL of statement
        User agent for SEC
    Returns:
        Free cash flow (int)
        Debt repayments (int)
        Amount spent on share buybacks (int)
        Amount spent on dividends (int)
        Share based compensation (int)
    '''

    # Get data from site
    content = requests.get(cf_url, headers=headers).content
    soup = BeautifulSoup(content, 'html.parser')

    # Initial values
    share_issue = buyback = divpaid = sbc = 0
    cfo = capex = fcf = debt_payment = '---'
    debt_pay = []

    # Find which column has 12 month data
    colm = column_finder_annual_htm(soup)

    # Loop through rows, search for row of interest
    for row in soup.table.find_all('tr'):
        tds = row.find_all('td')
        if ('Net cash from operations' in str(tds) or
            'this, \'defref_us-gaap_NetCashProvidedByUsedInOperatingActivities\', window' in str(tds) or
            'this, \'defref_us-gaap_NetCashProvidedByUsedInOperatingActivitiesContinuingOperations\', window' in str(tds)
            ):       
            cfo = html_re(str(tds[colm]))
            if cfo != '---':
                cfo = check_neg(str(tds), cfo)  
        elif ('Additions to property and equipment' in str(tds) or
              'this, \'defref_us-gaap_PaymentsToAcquireProductiveAssets\', window' in str(tds) or
              'this, \'defref_us-gaap_PaymentsToAcquirePropertyPlantAndEquipment\', window' in str(tds) or
              'this, \'defref_us-gaap_PaymentsForProceedsFromProductiveAssets\', window' in str(tds)
              ):
            capex = html_re(str(tds[colm]))
        elif ('this, \'defref_us-gaap_RepaymentsOfDebtMaturingInMoreThanThreeMonths\', window' in str(tds) or
              'this, \'defref_us-gaap_RepaymentsOfLongTermDebt\', window' in str(tds) or
              'this, \'defref_us-gaap_RepaymentsOfDebtAndCapitalLeaseObligations\', window' in str(tds) or
              'this, \'defref_us-gaap_InterestPaid\', window' in str(tds) or
              'this, \'defref_us-gaap_RepaymentsOfDebt\', window' in str(tds) or
              'RepaymentsOfShortTermAndLongTermBorrowings\', window' in str(tds) or
              'this, \'defref_us-gaap_RepaymentsOfLongTermDebtAndCapitalSecurities\', window' in str(tds) or
              'this, \'defref_us-gaap_InterestPaidNet\', window' in str(tds)
              ):
            debt_payment = html_re(str(tds[colm]))
            if debt_payment != '---':
                debt_pay.append(debt_payment)
        elif ('Common stock repurchased' in str(tds) or
              'this, \'defref_us-gaap_PaymentsForRepurchaseOfCommonStock\', window' in str(tds)
              ):
            buyback += html_re(str(tds[colm]))
        elif ('this, \'defref_us-gaap_ProceedsFromStockOptionsExercised\', window' in str(tds) or
              'this, \'defref_us-gaap_ProceedsFromIssuanceOfCommonStock\', window' in str(tds) or
              'this, \'defref_us-gaap_ProceedsFromIssuanceOfSharesUnderIncentiveAndShareBasedCompensationPlansIncludingStockOptions\', window' in str(tds)
              ):
            share_issue_rtn = html_re(str(tds[colm]))
            if share_issue_rtn != '---':
                share_issue += share_issue_rtn
        elif ('Common stock cash dividends paid' in str(tds) or
              'this, \'defref_us-gaap_PaymentsOfDividends\', window' in str(tds) or
              'PaymentsOfDividendsAndDividendEquivalentsOnCommonStockAndRestrictedStockUnits\', window' in str(tds) or
              'this, \'defref_us-gaap_PaymentsOfDividendsCommonStock\', window' in str(tds)
              ):
            divpaid = html_re(str(tds[colm]))
        elif ('this, \'defref_us-gaap_ShareBasedCompensation\', window' in str(tds) or
              'this, \'defref_us-gaap_AllocatedShareBasedCompensationExpense\', window' in str(tds)
              ):
            sbc = html_re(str(tds[colm]))

    # Calculate Free Cash Flow
    fcf = cfo - capex

    # Net out share issuance
    buyback -= share_issue

    # Sum debt payments, get rid of duplicates
    debt_pay = sum(set(debt_pay))

    return fcf, debt_pay, buyback, divpaid, sbc

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

def div_htm(div_url, headers):
    ''' Parse various .htm docs for div data

    Args:
        URL of statement
        User agent for SEC
    Returns:
        Dividend (float)
    '''

    # Get data from site
    content = requests.get(div_url, headers=headers).content
    soup = BeautifulSoup(content, 'html.parser')

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
            colm = 1
            # Check for three month data prior to 12 month data
            if '4 Months Ended' in str(soup):
                head = soup.table.find_all('tr')[0]
                
                # Check if 3 month data appears earlier
                four_mon = re.search(r'4 Months Ended', str(head), re.M)
                twelve_mon = re.search(r'12 Months Ended', str(head), re.M)
                if four_mon is not None and twelve_mon is not None:
                    if four_mon.span()[0] < twelve_mon.span()[0]:
                        # Find column where 12 month data starts
                        colm = re.findall(r'(?:colspan=\"(\d)\")(?!>12 Months Ended)', str(head), re.M)
                        colm = sum(map(int, colm))
            
            
            for row in soup.table.find_all('tr'):
                tds = row.find_all('td')
                if ('this, \'defref_us-gaap_CommonStockDividendsPerShareDeclared\', window' in str(tds) or
                    'this, \'defref_us-gaap_CommonStockDividendsPerShareCashPaid\', window' in str(tds)
                    ):
                    div = html_re(str(tds[colm]))
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

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# RE for .xml filings
def xml_re(string):
    ''' RE syntax and error checking for pulling numeric information from data cell on .xml filings

    Args:
        Target string (str)
    Return:
        Results (int/float)       
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


# Find correct column for data
def column_finder_annual_xml(soup):
    ''' Determine which column of data contains desired data for xml pages

    Args:
        Soup (soup data of sub filings)
    Returns:
        Column number (int)        
    '''

    # Check if needed to calculate
    if '12 Months Ended' in str(soup):        
        # Find correct column
        colm = re.search(r'(?:<Id>(\d\d?)</Id>\n)(?:<Labels>\n<Label Id=\"1\" Label=\"12 Months Ended\"/>)', str(soup), re.M)
        try:
            return int(colm.group(1)) - 1
        except:
            return 0
    else:
        return 0

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

def sum_xml(sum_url, headers):
    ''' Parse .xml document information
    
    Args:
        URL of statement
        User agent for SEC
    Returns:
        Fiscal year (int)
        Fiscal period end date (datetime object)        
    '''

    # Get data from site
    content = requests.get(sum_url, headers=headers).content
    soup = BeautifulSoup(content, 'xml')

    # Initial values
    fy = period_end = '---'

    # Loop through rows, search for FY and period end date
    rows = soup.find_all('Row')
    for row in rows:
        if r'DocumentFiscalYearFocus' in str(row):
            obj = re.search(r'(?:<NonNumbericText>)(\d+)(?:</NonNumbericText>)', str(row), re.M)
            fy = int(obj.group(1))
        elif r'dei_DocumentPeriodEndDate' in str(row):
            obj = re.search(r'(?:<NonNumbericText>)(.*?)(?:</NonNumbericText>)', str(row), re.M)
            period_end = datetime.strptime(obj.group(1), '%Y-%m-%d')  

    return fy, period_end

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

def rev_xml(rev_url, headers):
    ''' Parse .xml income statement

    Args:
        URL of statement
        User agent for SEC
    Returns:
        Revenue (int)
        Gross profit (int)
        R&D costs (int)
        Operating income (int)
        Net income (int)
        Earnings per share (float)
        Share count (int)
        Dividend (float)      
    '''    

    # Get data from site
    content = requests.get(rev_url, headers=headers).content
    soup = BeautifulSoup(content, 'xml')
    
    # Initial values
    rev = gross = oi = net = eps = cost = shares = div = '---'
    share_sum = research = 0

    # Find which column has 12 month data
    colm = column_finder_annual_xml(soup)

    # Loop through rows, search for row of interest
    rows = soup.find_all('Row')
    for row in rows:
        cells = row.find_all('Cell')
        if (r'<ElementName>us-gaap_Revenues</ElementName>' in str(row) or
            r'us-gaap_SalesRevenueNet' in str(row)
            ):
            rev = xml_re(str(cells[colm]))
        elif r'us-gaap_GrossProfit' in str(row):
            gross = xml_re(str(cells[colm]))
            if gross != '---':
                gross = check_neg(str(row), gross, 'xml')        
        elif (r'us-gaap_CostOfRevenue' in str(row) and gross == '---' or
              r'us-gaap_CostOfGoodsAndServicesSold' in str(row) and gross == '---' or
              r'us-gaap_CostOfGoodsSold' in str(row) and gross == '---'
              ):
            cost = xml_re(str(cells[colm]))              
        elif r'us-gaap_ResearchAndDevelopmentExpense' in str(row):
            research = xml_re(str(cells[colm]))         
        elif r'us-gaap_OperatingIncomeLoss' in str(row):
            oi = xml_re(str(cells[colm]))
            if oi != '---':
                oi = check_neg(str(row), oi, 'xml')      
        elif r'>us-gaap_NetIncomeLoss<' in str(row) and net == '---':
            net = xml_re(str(cells[colm]))
            if net != '---':
                net = check_neg(str(row), net, 'xml')                
        elif r'us-gaap_EarningsPerShareDiluted' in str(row) and eps == '---':
            eps = xml_re(str(cells[colm]))
            if eps != '---':
                eps = check_neg(str(row), eps, 'xml')
        elif r'us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding' in str(row):
            shares = xml_re(str(cells[colm]))
            share_sum += shares
        elif (r'us-gaap_CommonStockDividendsPerShareCashPaid' in str(row) or
              r'us-gaap_CommonStockDividendsPerShareDeclared' in str(row)):
            div = xml_re(str(cells[colm])) 

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

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

def bs_xml(bs_url, headers):
    ''' Parse .xml balance sheet statement

    Args:
        URL of statement
        User agent for SEC
    Returns:
        Cash (int)
        Assets minus goodwill (int)
        Long-term debt outstanding (int)
        Total liabilities (int)
        Shareholder's equity (int)
    '''   

    # Get data from site
    content = requests.get(bs_url, headers=headers).content
    soup = BeautifulSoup(content, 'xml')

    # Initial values
    equity = cash = assets = liabilities = '---'
    intangible_assets = goodwill = debt = 0

    # Find which column has 12 month data
    colm = column_finder_annual_xml(soup)

    # Loop through rows, search for row of interest
    rows = soup.find_all('Row')
    for row in rows:
        cells = row.find_all('Cell')
        if ('us-gaap_CashCashEquivalentsAndShortTermInvestments' in str(row) or
            'us-gaap_CashAndCashEquivalentsAtCarryingValue' in str(row)
            ):
            cash = xml_re(str(cells[colm])) 
        elif 'us-gaap_Goodwill' in str(row):
            goodwill = xml_re(str(cells[colm]))                
        elif r'us-gaap_IntangibleAssetsNetExcludingGoodwill' in str(row):
            intangible_assets = xml_re(str(cells[colm])) 
        elif r'<ElementName>us-gaap_Assets</ElementName>' in str(row):
            assets = xml_re(str(cells[colm]))       
        elif (r'us-gaap_LongTermDebtNoncurrent' in str(row) or
              r'us-gaap_LongTermDebtAndCapitalLeaseObligations' in str(row)
              ):
            debt = xml_re(str(cells[colm])) 
        elif r'<ElementName>us-gaap_Liabilities</ElementName>' in str(row):
            liabilities = xml_re(str(cells[colm])) 
        elif r'us-gaap_LiabilitiesAndStockholdersEquity' in str(row) and liabilities == '---':
            tot_liabilities = xml_re(str(cells[colm])) 
        elif r'<ElementName>us-gaap_StockholdersEquity</ElementName>' in str(row):
            equity = xml_re(str(cells[colm])) 
            if equity != '---':
                equity = check_neg(str(row), equity, 'xml')                                                 

    # Calculate liabilites from shareholder equity if not found
    if liabilities == '---' and tot_liabilities != '---' and equity != '---':
        liabilities = tot_liabilities - equity

    # Net out goodwill from assets
    if assets != '---' and goodwill != '---':
        assets -= (goodwill + intangible_assets)
    
    return cash, assets, debt, liabilities, equity

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

def cf_xml(cf_url, headers):    
    ''' Parse .htm cash flow statement

    Args:
        URL of statement
        User agent for SEC
    Returns:
        Free cash flow (int)
        Debt repayments (int)
        Amount spent on share buybacks (int)
        Amount spent on dividends (int)
        Share based compensation (int)
    '''

    # Get data from site
    content = requests.get(cf_url, headers=headers).content
    soup = BeautifulSoup(content, 'xml')

    # Initial values
    debt_pay = share_issue = buyback = divpaid = sbc = 0
    cfo = capex = fcf = '---'

    # Find which column has 12 month data
    colm = column_finder_annual_xml(soup)

    # Loop through rows, search for row of interest
    rows = soup.find_all('Row')
    for row in rows:
        cells = row.find_all('Cell')
        if r'<ElementName>us-gaap_NetCashProvidedByUsedInOperatingActivities</ElementName>' in str(row):
            cfo = xml_re(str(cells[colm]))
            if cfo != '---':
                cfo = check_neg(str(row), cfo, 'xml')
        elif (r'us-gaap_PaymentsToAcquirePropertyPlantAndEquipment' in str(row) or
              r'us-gaap_PaymentsToAcquireProductiveAssets' in str(row)
              ):
            capex = xml_re(str(cells[colm]))
        elif (r'us-gaap_ProceedsFromRepaymentsOfShortTermDebtMaturingInThreeMonthsOrLess' in str(row) or
              r'RepaymentsOfShortTermAndLongTermBorrowings' in str(row) or
              r'us-gaap_RepaymentsOfLongTermDebtAndCapitalSecurities' in str(row) or
              r'us-gaap_InterestPaid' in str(row) or
              r'us-gaap_RepaymentsOfLongTermDebt' in str(row)
              ):
            debt_pay += xml_re(str(cells[colm]))                             
        elif r'us-gaap_PaymentsForRepurchaseOfCommonStock' in str(row):
            buyback += xml_re(str(cells[colm]))     
        elif (r'us-gaap_ProceedsFromStockOptionsExercised' in str(row) or
              r'us-gaap_ProceedsFromIssuanceOfCommonStock' in str(row) or
              r'cost_ProceedsFromStockbasedAwardsNet' in str(row)
              ):
            share_issue = xml_re(str(cells[colm]))   
        elif (r'us-gaap_PaymentsOfDividendsCommonStock' in str(row) or
              r'us-gaap_PaymentsOfDividends' in str(row)
              ):
            divpaid = xml_re(str(cells[colm]))            
        elif r'us-gaap_ShareBasedCompensation' in str(row):
            sbc = xml_re(str(cells[colm]))
    
    # Calculate Free Cash Flow
    fcf = cfo - capex

    # Net out share issuance
    buyback -= share_issue

    return fcf, debt_pay, buyback, divpaid, sbc

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

def div_xml(div_url, headers):
    ''' Parse various .htm docs for div data

    Args:
        URL of statement
        User agent for SEC
    Returns:
        Dividend (float)
    '''

    # Remove for query use
    content = requests.get(div_url, headers=headers).content
    soup = BeautifulSoup(content, 'xml')

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
        if (r'>us-gaap_CommonStockDividendsPerShareDeclared' in str(row) or
            r'us-gaap_CommonStockDividendsPerShareCashPaid' in str(row)
            ):
            cells = row.find_all('Cell')
            for cell in cells:
                if search_id in str(cell):
                    div = xml_re(str(cell))
                    break
    
    # Check if reporting quarterly div
    if 'QUARTERLY' in str(row).upper():
        div *= 4

    return div