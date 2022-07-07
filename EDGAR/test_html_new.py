import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import date
from datetime import datetime
import pandas as pd
import math

with open(r'C:\Users\unit0\OneDrive\Desktop\EDGAR\user_agent.txt') as f:
    data = f.read()
    headers = json.loads(data)


# RE for .htm filings
def html_re(string):
    ''' RE syntax and error checking for pulling numeric information from data cell on .htm filings

    Args:
        Target string (str)
    Return:
        results (int/float)       
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
def column_finder_annual_htm(soup):
    ''' Determine which column of data contains desired data for htm pages

    Args:
        soup (soup data of sub filings)
    Returns:
        column number (int)        
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


url = r'https://www.sec.gov/Archives/edgar/data/789019/000156459020034944/FilingSummary.xml'
sum_url = r'https://www.sec.gov/Archives/edgar/data/1403161/000140316117000044/R1.htm'

def sum_htm_test(sum_url):

    # Remove for query use
    content = requests.get(sum_url, headers=headers).content
    soup = BeautifulSoup(content, 'html.parser')

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
        if r"this, 'defref_dei_DocumentPeriodEndDate', window" in str(tds):
            period_end_str = tds[1].text.strip()
            period_end = datetime.strptime(period_end_str, '%b. %d,  %Y') 

        return fy, period_end

# Sum Test
#print(sum_htm_test(sum_url))


'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# rev, gross, research, oi, net, eps, share_sum, div
rev_url_list = [r'https://www.sec.gov/Archives/edgar/data/789019/000156459021039151/R2.htm', r'https://www.sec.gov/Archives/edgar/data/320193/000032019320000096/R2.htm', r'https://www.sec.gov/Archives/edgar/data/320193/000162828016020309/R2.htm', 
                r'https://www.sec.gov/Archives/edgar/data/320193/000119312513416534/R2.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000156459017014900/R2.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000119312513310206/R2.htm',
                r'https://www.sec.gov/Archives/edgar/data/1018724/000101872421000004/R3.htm', r'https://www.sec.gov/Archives/edgar/data/1018724/000101872418000005/R3.htm', r'https://www.sec.gov/Archives/edgar/data/1018724/000101872414000006/R3.htm',
                r'https://www.sec.gov/Archives/edgar/data/1652044/000165204421000010/R4.htm', r'https://www.sec.gov/Archives/edgar/data/1652044/000165204419000004/R4.htm', r'https://www.sec.gov/Archives/edgar/data/1652044/000165204416000012/R4.htm',
                r'https://www.sec.gov/Archives/edgar/data/1403161/000140316120000070/R4.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000140316117000044/R4.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000140316114000017/R4.htm',
                r'https://www.sec.gov/Archives/edgar/data/789019/000156459019027952/R2.htm', r'https://www.sec.gov/Archives/edgar/data/320193/000119312512444068/R2.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000119312515272806/R2.htm',
                r'https://www.sec.gov/Archives/edgar/data/320193/000119312514383437/R2.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000119312512316848/R2.htm', r'https://www.sec.gov/Archives/edgar/data/320193/000032019318000145/R2.htm', 
                r'https://www.sec.gov/Archives/edgar/data/1652044/000165204418000007/R4.htm', r'https://www.sec.gov/Archives/edgar/data/354950/000035495015000008/R4.htm', r'https://www.sec.gov/Archives/edgar/data/1318605/000156459017003118/R4.htm',
                r'https://www.sec.gov/Archives/edgar/data/1318605/000119312514069681/R4.htm', r'https://www.sec.gov/Archives/edgar/data/909832/000090983220000017/R2.htm', r'https://www.sec.gov/Archives/edgar/data/909832/000144530513002422/R4.htm',
                r'https://www.sec.gov/Archives/edgar/data/909832/000119312512428890/R4.htm']
rev_answers = [[168088, 115856, 20716, 69916, 61271, 8.05, 7608, '---'], [274515, 104956, 18752, 66288, 57411, 3.28, 17528214, '---'], [215639, 84263, 10045, 60024, 45687, 8.31, 5500281, 2.18],
               [170910, 64304, 4475, 48999, 37037, 39.75, 931662, 11.4], [89950, 55689, 13037, 22326, 21204, 2.71, 7832, 1.56], [77849, 57600, 10411, 26764, 21863, 2.58, 8470, 0.92],
               [386064, 152757, 42740, 22899, 21331, 41.83, 510, '---'], [177866, 65932, 22620, 4106, 3033, 6.15, 493, '---'], [74452, 20271, 6565, 745, 274, 0.59, 465, '---'],
               [182527, 97795, 27573, 41224, 40269, 58.61, 687, '---'], [136819, 77270, 21419, 26321, 30736, 43.7, 703, '---'], [74989, 46825, 12282, 19360, 16348, 22.84, 716, '---'],
               [21846, 21846, 0, 14081, 10866, 4.89, 2479, '---'], [18358, 18358, 0, 12144, 6699, 2.8, 2654, '---'], [12702, 12702, 0, 7697, 5438, 8.62, 902, '---'],
               [125843, 82933, 16876, 42959, 39240, 5.06, 7753, '---'], [156508, 68662, 3381, 55241, 41733, 44.15, 945355, 2.65], [93580, 60542, 12046, 18161, 12193, 1.48, 8254, 1.24],
               [182795, 70537, 6041, 52503, 39510, 6.45, 6122663, 1.82], [73723, 56193, 9811, 21763, 16978, 2.0, 8506, 0.8], [265595, 101839, 14236, 70898, 59531, 11.91, 5000109, '---'],
               [110855, 65272, 16625, 26146, 12662, 18.0, 703, '---'], [83176, 28954, 0, 10469, 6345, 4.71, 1346, '---'], [7000132, 1599257, 834408, -667340, -674914, -4.68, 144212, '---'],
               [2013496, 456262, 231976, -61283, -74014, -0.62, 119421414, '---'], [166761, 21822, 0, 5435, 4002, 9.02, 443901, '---'], [105156, 13208, 0, 3053, 2039, 4.63, 440512, 8.17],
               [99137, 12314, 0, 2759, 1709, 3.89, 439373, 1.03]]


rev_url = r'https://www.sec.gov/Archives/edgar/data/1318605/000156459017003118/R4.htm'


def rev_htm_test(rev_url):
    
    # Remove for query use
    content = requests.get(rev_url, headers=headers).content
    soup = BeautifulSoup(content, 'html.parser')
    
    ''' Parse .htm income statement

    Args:
        soup (soup data of sub filings)
    Returns:
        revenue (int), gross profit (int), R&D costs (int), operating income (int), net income (int), earnings per share (float), share count (int), dividend (float)      
    '''

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

# Rev Test
#print(rev_htm_test(rev_url))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# cash, assets, debt, liabilities, equity
bs_url_list = [r'https://www.sec.gov/Archives/edgar/data/1403161/000140316120000070/R2.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000140316118000055/R2.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000140316116000058/R2.htm',
               r'https://www.sec.gov/Archives/edgar/data/1403161/000140316114000017/R2.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000119312511315956/R2.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000156459021039151/R4.htm', 
               r'https://www.sec.gov/Archives/edgar/data/789019/000156459019027952/R4.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000156459017014900/R5.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000119312515272806/R5.htm',
               r'https://www.sec.gov/Archives/edgar/data/789019/000119312513310206/R5.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000119312512316848/R3.htm', r'https://www.sec.gov/Archives/edgar/data/320193/000032019320000096/R4.htm',
               r'https://www.sec.gov/Archives/edgar/data/320193/000032019317000070/R5.htm', r'https://www.sec.gov/Archives/edgar/data/320193/000119312514383437/R5.htm', r'https://www.sec.gov/Archives/edgar/data/320193/000119312511282113/R3.htm',
               r'https://www.sec.gov/Archives/edgar/data/1652044/000165204421000010/R2.htm', r'https://www.sec.gov/Archives/edgar/data/1652044/000165204418000007/R2.htm', r'https://www.sec.gov/Archives/edgar/data/1652044/000165204416000012/R2.htm',
               r'https://www.sec.gov/Archives/edgar/data/354950/000035495017000005/R2.htm', r'https://www.sec.gov/Archives/edgar/data/354950/000035495020000015/R2.htm', r'https://www.sec.gov/Archives/edgar/data/909832/000090983220000017/R4.htm',
               r'https://www.sec.gov/Archives/edgar/data/909832/000090983218000013/R2.htm', r'https://www.sec.gov/Archives/edgar/data/77476/000007747621000007/R5.htm']
bs_answers = [[16289, 37201, 21071, 44709, 36210], [8162, 26473, 16630, 35219, 34006], [5619, 21735, 15882, 31123, 32912],
              [1971, 15405, 0, 11156, 27413], [2127, 11656, 0, 8323, 26437], [14224, 276268, 50074, 191791, 141988],
              [11356, 236780, 66662, 184226, 102330], [7663, 195858, 76073, 168692, 72394], [5595, 154449, 27808, 96140, 80083],
              [3804, 124693, 12601, 63487, 78944], [6938, 104649, 10713, 54908, 66363], [38016, 323888, 98667, 258549, 65339],
              [20289, 367304, 97207, 241272, 134047], [13844, 223081, 28987, 120292, 111547], [9815, 111939, 0, 39756, 76615],
              [26465, 296996, 13932, 97072, 222544], [10715, 177856, 3969, 44793, 152502], [16549, 127745, 1995, 27130, 120331],
              [2538, 40873, 22349, 38633, 4333], [2133, 48982, 28670, 54352, -3116], [12277, 55556, 7514, 36851, 18284],
              [6055000000, 40830000000, 6487000000, 27727000000, 12799000000], [8185, 54846, 40370, 79366, 13454]]

bs_url = r'https://www.sec.gov/Archives/edgar/data/1652044/000165204416000012/R2.htm'


def bs_htm_test(bs_url):
    
    # Remove for query use
    content = requests.get(bs_url, headers=headers).content
    soup = BeautifulSoup(content, 'html.parser')
    
    ''' Parse .htm balance sheet statement

    Args:
        soup (soup data of sub filings)
    Returns:
        cash (int), assets minus goodwill (int), long-term debt outstanding (int), total liabilities (int), shareholder's equity (int)
    '''
    
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

# BS Test
#print(bs_htm_test(bs_url))


'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# fcf, debt_pay, buyback, divpaid, sbc
cf_url_list = [r'https://www.sec.gov/Archives/edgar/data/1652044/000165204418000007/R8.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000140316120000070/R8.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000140316118000055/R8.htm',
               r'https://www.sec.gov/Archives/edgar/data/1403161/000140316116000058/R8.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000140316114000017/R8.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000138410812000011/R8.htm',
               r'https://www.sec.gov/Archives/edgar/data/789019/000156459021039151/R6.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000156459019027952/R6.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000156459017014900/R7.htm',
               r'https://www.sec.gov/Archives/edgar/data/789019/000119312515272806/R7.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000119312513310206/R7.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000119312512316848/R5.htm',
               r'https://www.sec.gov/Archives/edgar/data/320193/000032019320000096/R7.htm', r'https://www.sec.gov/Archives/edgar/data/320193/000032019317000070/R8.htm', r'https://www.sec.gov/Archives/edgar/data/320193/000119312514383437/R8.htm',
               r'https://www.sec.gov/Archives/edgar/data/320193/000119312511282113/R6.htm', r'https://www.sec.gov/Archives/edgar/data/1652044/000165204421000010/R8.htm', r'https://www.sec.gov/Archives/edgar/data/1652044/000165204416000012/R8.htm',
               r'https://www.sec.gov/Archives/edgar/data/789019/000119312511200680/R5.htm', r'https://www.sec.gov/Archives/edgar/data/354950/000035495021000089/R7.htm', r'https://www.sec.gov/Archives/edgar/data/354950/000035495015000008/R8.htm',
               r'https://www.sec.gov/Archives/edgar/data/1318605/000156459019003165/R8.htm', r'https://www.sec.gov/Archives/edgar/data/1318605/000156459016013195/R8.htm', r'https://www.sec.gov/Archives/edgar/data/909832/000144530513002422/R7.htm',
               r'https://www.sec.gov/Archives/edgar/data/1538990/000155837021002003/R8.htm']
cf_answers = [[23907, 4461, 4846, 0, 7679], [9704, 537, 7924, 2664, 416], [11995, 2295, 7028, 1918, 327],
              [5051, 244, 7062, 1350, 221], [6652, 0, 4027, 1006, 172], [4633, 0, 536, 595, 147],
              [56118, 3750, 25692, 16521, 6118], [38260, 4000, 18401, 13811, 4652], [31378, 7922, 11016, 11845, 3266],
              [23136, 1500, 13809, 9882, 2574], [24576, 1346, 4429, 7455, 2406], [29321, 0, 3116, 6385, 2244],
              [73365, 15631, 71478, 14081, 6829], [51147, 5592, 32345, 12769, 4840], [50142, 339, 44270, 11126, 2863],
              [33269, 0, -831, 0, 1168], [42843, 2100, 31149, 0, 12991], [16109, 13824, 1780, 47, 5203],
              [24639, 814, 9133, 5180, 2166], [16376, 4113, 465, 6451, 310], [6800, 821, 6748, 2530, 225],
              [-2922, 380836, -295722, 0, 749024],[-2159349, 32060, -836611, 0, 197999], [1354, 86, -16, 3560, 285],
              []]

cf_url = r'https://www.sec.gov/Archives/edgar/data/1538990/000155837021002003/R8.htm'

def cf_htm_test(cf_url):
    
    # Remove for query use
    content = requests.get(cf_url, headers=headers).content
    soup = BeautifulSoup(content, 'html.parser')

    ''' Parse .htm cash flow statement

    Args:
        soup (soup data of sub filings)
    Returns:
        free cash flow (int), debt repayments (int), amount spent on share buybacks (int), amount spent on dividends (int), share based compensation (int)
    '''

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

# CF Test
#print(cf_htm_test(cf_url))


'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


div_url_list = [r'https://www.sec.gov/Archives/edgar/data/1403161/000140316120000070/R7.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000140316119000050/R7.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000140316118000055/R7.htm',
                r'https://www.sec.gov/Archives/edgar/data/1403161/000140316115000013/R7.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000138410812000011/R7.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000140316114000017/R7.htm',
                r'https://www.sec.gov/Archives/edgar/data/1403161/000140316116000058/R7.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000156459021039151/R99.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000156459019027952/R107.htm',
                r'https://www.sec.gov/Archives/edgar/data/789019/000156459017014900/R110.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000119312515272806/R112.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000119312513310206/R102.htm',
                r'https://www.sec.gov/Archives/edgar/data/789019/000119312512316848/R104.htm', r'https://www.sec.gov/Archives/edgar/data/320193/000032019320000096/R6.htm', r'https://www.sec.gov/Archives/edgar/data/320193/000032019318000145/R8.htm',
                r'https://www.sec.gov/Archives/edgar/data/354950/000035495021000089/R13.htm', r'https://www.sec.gov/Archives/edgar/data/354950/000035495017000005/R7.htm', r'https://www.sec.gov/Archives/edgar/data/354950/000035495013000008/R7.htm',
                r'https://www.sec.gov/Archives/edgar/data/909832/000090983220000017/R43.htm']
div_answers = [1.2, 1.0, 0.825,
               0.48, 0.88, 1.6,
               0.56, 2.24, 1.84,
               1.56, 1.24, 0.92,
               0.8, 0.795, 2.72,
               6.0, 2.76, 1.16,
               2.7]

div_url = r'https://www.sec.gov/Archives/edgar/data/77476/000007747621000007/R6.htm'

def div_htm_test(div_url):
    
    # Remove for query use
    content = requests.get(div_url, headers=headers).content
    soup = BeautifulSoup(content, 'html.parser')
    
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
                    div = html_re(str(tds), colm)
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

# Div Test
#print(div_htm_test(div_url))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

Earnings_Per_Share = [[3.28, 11.89, 11.91, 9.21, 8.31, 9.22, 6.45, 39.75, 44.15, 27.68, 15.15], [4.89, 5.32, 4.42, 2.8, 2.48, 2.58, 8.62, 7.59, 3.16, 5.16, 4.01],
                      [0.64, -0.7, -0.82, -1.69, -0.67, -0.99, -0.34, -0.09, -0.53, -0.0, -0.0]]
Shares_Outstanding = [[17528214, 4648913, 5000109, 5251692, 5500281, 5793069, 6122663, 931662, 945355, 936645, 924712], [2479, 2529, 2586, 2654, 2678, 2724, 902, 929, 964, 1022, 1096],
                      [1083, 1239, 1193675, 1160306, 1009484, 897414, 871775401, 835949898, 751444316, 751771608, 751771608]]
Dividends = [[0.68, 2.4, 2.72, 2.4, 2.18, 1.98, 1.82, 11.4, 2.65, '---', '---'], [1.2, 1.0, 0.825, 0.66, 0.56, 0.48, 1.6, 1.32, 0.88, 0.6, 0.5],
             ['---', '---', '---', '---', '---', '---', '---', '---', '---', '---', '---']]
index = 2           # Change me!
eps = Earnings_Per_Share[index]
shares = Shares_Outstanding[index]
div = Dividends[index]

def split_test(Earnings_Per_Share, Shares_Outstanding, Dividends):
    ''' Adjust share and per share metrics based on stock splits and mutltiples changes

    Args:
        EPS (list), shares outstanding (list), dividends (list)
    Returns:
        split adjusted EPS (list), adjusted shares outstanding (list), split adjusted dividends (list)
    '''
    # Determine if share split occured, and adjust per share metrics
    split_factor = 1
    adjusted_shares = [Shares_Outstanding[0]]

    # Loop through share count and look for major differences
    for i in range(1, len(Shares_Outstanding)):
        share_ratio = (Shares_Outstanding[i-1] / Shares_Outstanding[i])
        
        # If difference found, update split factor
        if share_ratio >= 1.45 or share_ratio <= 0.70:
            split_factor *= math.ceil(share_ratio)
        
        # If there have been splits, adjust per share metrics
        if split_factor != 1:
            Earnings_Per_Share[i] = round(Earnings_Per_Share[i] / split_factor, 2)
            if Dividends[i] != '---':
                Dividends[i] = round(Dividends[i] / split_factor, 3)
    
        # Append shares to adjusted shares
        adjusted_shares.append(Shares_Outstanding[i] * split_factor)    
    
    return Earnings_Per_Share, adjusted_shares, Dividends



multi_values = [[51628, 33772, 29388963, 28233633, 22287931, 8092460, 5849251, 2416930, 1114190, 713448, 713448], [31536, 24578, 21461268, 11758751, 7000132, 4046025, 3198356, 2013496, 413256, 204242, 204242],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 9403672, 9415700, 5860049, 2040375, 0, 0, 401495, 268335, 268335], [55556, 45400, 40830000000, 36347000000, 33163, 33440, 33024, 30283, 27140, 26761, 23815],
                [196, 247, 328, 469, 486, 481, 296, -16, 523, 400, 358]]
multi_answers = []

multi_index = 5
values = multi_values[multi_index]

def multiplier_test(values):
    ''' Adjust numbers based on multplier change

    Args:
        values (list)
    Returns:
        results (list)
    '''

    multiplier_change = False
    # Loop through share count and look for major differences
    for i in range(1, len(values)):
        if values[i] != 0:
            ratio = (values[i-1] / values[i])
        
            # Check if multiplier (thousands, millions) was changed
            if ratio >= 900 or ratio <= .0015:
                multiplier_change = True
                break

    # Adjust for multiplier change
    if multiplier_change == True:
        multi_factor = 1
        results = [values[0]]
        
        # Determine where there is a multiplier change
        for i in range(1, len(values)):
            if values[i] != 0 and values[i-1] != 0:
                ratio = abs(values[i-1] / values[i])
                if ratio > 800_000:
                    multi_factor *= 1_000_000
                elif ratio >= 800 and ratio < 1200:
                    multi_factor *= 1000
                elif ratio <= 0.0015 and ratio > 0.00001:
                    multi_factor /= 1000
                elif ratio < .000005:
                    multi_factor /= 1_000_000
            
            # Adjust for multiplier
            results.append(int(values[i] * multi_factor))

        return results

    return values


# Split Test
#Earnings_Per_Share, Adjusted_Shares, Dividends = split_test(eps, shares, div)
#Adjusted_Shares = multiplier_test(Adjusted_Shares)
#print(Earnings_Per_Share)
#print(Dividends)
#print(Adjusted_Shares)
#print(multiplier_test(values))
