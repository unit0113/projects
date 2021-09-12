import requests
from bs4 import BeautifulSoup
import json
import re
import xml.etree.ElementTree as ET
from datetime import date
from datetime import datetime

with open(r'C:\Users\unit0\OneDrive\Desktop\EDGAR\user_agent.txt') as f:
    data = f.read()
    headers = json.loads(data)

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

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


# Check for possible negative number when not expected
def check_neg(string, value, type='htm'):
    ''' Checks for unexpected negative numbers

    Args:
        Target string (str), value that we're checking (int/float)
    Return:
        value (int/float)       
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
def column_finder_annual_xml(soup):
    ''' Determine which column of data contains desired data for xml pages

    Args:
        soup (soup data of sub filings)
    Returns:
        column number (int)        
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

sum_url = r'https://www.sec.gov/Archives/edgar/data/1018724/000119312511016253/R1.xml'

def sum_xml_test(sum_url):
    
    # Remove for query use
    content = requests.get(sum_url, headers=headers).content
    soup = BeautifulSoup(content, 'xml')
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
        if r'DocumentFiscalYearFocus' in str(row):
            obj = re.search(r'(?:<NonNumbericText>)(\d+)(?:</NonNumbericText>)', str(row), re.M)
            fy = int(obj.group(1))
        elif r'dei_DocumentPeriodEndDate' in str(row):
            obj = re.search(r'(?:<NonNumbericText>)(.*?)(?:</NonNumbericText>)', str(row), re.M)
            period_end = datetime.strptime(obj.group(1), '%Y-%m-%d')  

    return fy, period_end     
#print(sum_xml_test(sum_url))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

rev_url_list = [r'https://www.sec.gov/Archives/edgar/data/1403161/000119312510265236/R4.xml', r'https://www.sec.gov/Archives/edgar/data/320193/000119312510238044/R2.xml', r'https://www.sec.gov/Archives/edgar/data/909832/000119312510230379/R4.xml']
rev_answers = [[8065, 8065, 0, 4589, 2966, 4.01, 1096, '---'], [65225, 25684, 1782, 18385, 14013, 15.15, 924712, '---'], [77946, 9951, 0, 2077, 1303, 2.92, 445970, 0.77]]

rev_url = r'https://www.sec.gov/Archives/edgar/data/909832/000119312510230379/R4.xml'


def rev_xml_test(rev_url):
    
    # Remove for query use
    content = requests.get(rev_url, headers=headers).content
    soup = BeautifulSoup(content, 'xml')

    ''' Parse .xml income statement

    Args:
        soup (soup data of sub filings)
    Returns:
        revenue (int), gross profit (int), R&D costs (int), operating income (int), net income (int), earnings per share (float), share count (int), dividend (float)      
    '''    
    
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
        elif r'us-gaap_CommonStockDividendsPerShareCashPaid' in str(row):
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

#print(rev_xml_test(rev_url))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

bs_url_list = [r'https://www.sec.gov/Archives/edgar/data/1403161/000119312510265236/R2.xml', r'https://www.sec.gov/Archives/edgar/data/1018724/000119312511016253/R5.xml', r'https://www.sec.gov/Archives/edgar/data/320193/000119312510238044/R3.xml',
               r'https://www.sec.gov/Archives/edgar/data/354950/000119312511076501/R3.xml', r'https://www.sec.gov/Archives/edgar/data/909832/000119312510230379/R2.xml']
bs_answers = [[3867, 10483, 32, 8394, 25011], [3777, 17448, 0, 11933, 6864], [11261, 74100, 0, 27392, 47791],
              [545, 38938, 8707, 21236, 18889], [3214, 23815, 2141, 12885, 10829]]

bs_url = r'https://www.sec.gov/Archives/edgar/data/1018724/000119312511016253/R5.xml'

def bs_xml_test(bs_url):
    
    # Remove for query use
    content = requests.get(bs_url, headers=headers).content
    soup = BeautifulSoup(content, 'xml')

    ''' Parse .xml balance sheet statement

    Args:
        soup (soup data of sub filings)
    Returns:
        cash (int), assets minus goodwill (int), long-term debt outstanding (int), total liabilities (int), shareholder's equity (int)
    '''   

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

#print(bs_xml_test(bs_url))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

cf_url_list = [r'https://www.sec.gov/Archives/edgar/data/1403161/000119312510265236/R8.xml', r'https://www.sec.gov/Archives/edgar/data/320193/000119312510238044/R6.xml', r'https://www.sec.gov/Archives/edgar/data/909832/000119312510230379/R7.xml']
cf_answers = [[2450, 16, 944, 368, 131], [16590, 0, -912, 0, 879], [1725, 194, 358, 338, 190]]


cf_url = r'https://www.sec.gov/Archives/edgar/data/909832/000119312510230379/R7.xml'

def cf_xml_test(cf_url):
    
    # Remove for query use
    content = requests.get(cf_url, headers=headers).content
    soup = BeautifulSoup(content, 'xml')
    
    ''' Parse .htm cash flow statement

    Args:
        soup (soup data of sub filings)
    Returns:
        free cash flow (int), debt repayments (int), amount spent on share buybacks (int), amount spent on dividends (int), share based compensation (int)
    '''

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

# CF Test
print(cf_xml_test(cf_url))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

div_url_list = [r'https://www.sec.gov/Archives/edgar/data/789019/000119312510171791/R101.xml', r'https://www.sec.gov/Archives/edgar/data/1403161/000119312510265236/R6.xml', r'https://www.sec.gov/Archives/edgar/data/354950/000119312511076501/R6.xml']
div_answers = [0.52, 0.5, 0.945]

div_url = r'https://www.sec.gov/Archives/edgar/data/354950/000119312511076501/R6.xml'

def div_xml_test(div_url):
    
    # Remove for query use
    content = requests.get(div_url, headers=headers).content
    soup = BeautifulSoup(content, 'xml')

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

# Div test
#print(div_xml_test(div_url))
