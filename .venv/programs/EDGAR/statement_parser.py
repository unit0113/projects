from pandas.io.pytables import Table
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
        if isinstance(value, int):
            re_str = '.' + "{:,}".format(value) + '.'
        else:
            re_str = '.' + format(value, '.2f') + '.'
    else:
        re_str = '.' + str(value) + '.'
    
    obj = re.search(re_str, string, re.M)
    
    # Check for weird float
    if obj == None:
        re_str = '.' + "{:,}".format(value, '.1f') + '.'
        obj = re.search(re_str, string, re.M)

    # Check if negative
    if '(' in obj.group(0) and ')' in obj.group(0):
        return (value * -1)
    else:
        return value


# Find correct column for data
def column_finder_annual_htm(soup, per):
    ''' Determine which column of data contains desired data for htm pages

    Args:
        Soup (soup data of sub filings)
    Returns:
        Column number (int)        
    '''

    # Extract colmn header string
    head = soup.table.find_all('tr')[0]
    head2 = soup.table.find_all('tr')[1]

    # Find index for period end
    if per in str(head2):
        rows = head2.find_all('th')
        for index, row in enumerate(rows):
            if per in str(row) and '3 Months Ended' not in str(row) and '4 Months Ended' not in str(row):
                # Find correct column if multiple things before 12 months data
                col_index = 0
                colm = 0
                rows2 = head.find_all('th')
                for row in rows2:
                    if col_index > index:
                        if '3 Months Ended' in str(row) or '4 Months Ended' in str(row):
                            break 
                        else:
                            return colm           
                    if 'colspan' in str(row):
                        colm_part = re.search(r'(?:colspan=\"(\d\d?)\")', str(row), re.M)
                        colm += int(colm_part.group(1))
                        col_index += int(colm_part.group(1))
    else:
        # Find correct column accounting for empties prior to data
        colm = re.search(r'(?:colspan=\"(\d\d?)\")', str(head), re.M)
        return int(colm.group(1))


multiplier_list_1 = ['shares in Millions, $ in Millions', 'In Millions, except Per Share data, unless otherwise specified', 'In Millions, except Per Share data']

multiplier_list_2 = ['shares in Thousands, $ in Millions', 'In Millions, except Share data in Thousands, unless otherwise specified']

multiplier_list_3 = ['shares in Thousands, $ in Thousands', 'In Thousands, except Per Share data, unless otherwise specified', 'In Thousands, except Per Share data']

multiplier_list_4 = ['In Thousands, except Share data, unless otherwise specified', 'In Thousands, except Share data']

multiplier_list_5 = ['$ in Millions', 'In Millions, unless otherwise specified', 'In Millions']

multiplier_list_6 = ['$ in Thousands', 'In Thousands, unless otherwise specified', 'In Thousands']

multiplier_list_7 = ['shares in Thousands']

multiplier_list_8 = ['shares in Millions']


def multiple_extractor(head, shares=False, xml=False):
    
    # Search for multiplier string
    if xml == True:
        obj = re.search(r'(?:<RoundingOption>)(.*)(?:</RoundingOption>)', head)
    else: 
        obj = re.search(r'(?:<br/?>)(.*)(?:</strong>)', head)
    
    result = obj.group(1).strip()

    # Determine multiplier
    if result in multiplier_list_1:
        if shares == True:
            return 1_000_000, 1_000_000
        else:
            return 1_000_000
    elif result in multiplier_list_2 and shares == True:
        return 1_000_000, 1_000
    elif result in multiplier_list_3:
        if shares == True:
            return 1_000, 1_000
        else:
            return 1_000
    elif result in multiplier_list_4 and shares == True:
        return 1_000, 1
    elif result in multiplier_list_5:
        if shares == True:
            return 1_000_000, 1_000_000
        else:
            return 1_000_000
    elif result in multiplier_list_6:
        if shares == True:
            return 1_000, 1
        else:
            return 1_000    
    elif result in multiplier_list_7 and shares == True:
        return 1, 1_000
    elif result in multiplier_list_8 and shares == True:
        return 1, 1_000_000
    else:
        if shares == True:
            return 1, 1
        else:
            return 1


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
            period_end = period_end_str.replace('  ', ' ')

    return fy, period_end


'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


def rev_htm(rev_url, headers, per):    
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
    gross = oi = net = eps = cost = shares = div = '---'
    rev = share_sum = research = non_attributable_net = dep_am = impairment = disposition = ffo = operating_exp = 0
    share_set = set()
    net_actual = False

    # Find which column has 12 month data
    colm = column_finder_annual_htm(soup, per)

    # Determine multiplier
    head = soup.find('th')
    dollar_multiplier, share_multiplier = multiple_extractor(str(head), shares=True)
 
    # Loop through rows, search for row of interest
    for row in soup.table.find_all('tr'):
        tds = row.find_all('td')
        if (r"this, 'defref_us-gaap_Revenues', window" in str(tds) or
            r"this, 'defref_us-gaap_SalesRevenueNet', window" in str(tds) or
            r'defref_us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax' in str(tds) or
            r"this, 'defref_us-gaap_SalesRevenueGoodsNet', window" in str(tds) or
            r"this, 'defref_us-gaap_RealEstateRevenueNet', window" in str(tds) or
            r"this, 'defref_us-gaap_RevenueFromContractWithCustomerIncludingAssessedTax', window" in str(tds)
            ):
            rev_calc = html_re(str(tds[colm]))
            if rev_calc != '---':
                rev_calc = round(rev_calc * (dollar_multiplier / 1_000_000))
                if rev_calc > rev:
                    rev = rev_calc
        elif r'defref_us-gaap_GrossProfit' in str(tds):
            gross = html_re(str(tds[colm]))
            if gross != '---':
                gross = round(check_neg(str(tds), gross) * (dollar_multiplier / 1_000_000))
        elif (r"this, 'defref_us-gaap_ResearchAndDevelopmentExpense', window" in str(tds) or
              r"this, 'defref_amzn_TechnologyAndContentExpense', window" in str(tds) or
              r"this, 'defref_us-gaap_ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost', window" in str(tds)
              ):
            research = round(html_re(str(tds[colm])) * (dollar_multiplier / 1_000_000))
        elif r"this, 'defref_us-gaap_OperatingIncomeLoss', window" in str(tds):
            oi = html_re(str(tds[colm]))
            if oi != '---':
                oi = round(check_neg(str(tds), oi) * (dollar_multiplier / 1_000_000))
        elif ('this, \'defref_us-gaap_NetIncomeLoss\', window' in str(tds) and net == '---' or
              'this, \'defref_us-gaap_ProfitLoss\', window' in str(tds) and net == '---' or
              r"this, 'defref_us-gaap_NetIncomeLossAvailableToCommonStockholdersBasic', window" in str(tds) and net == '---'
            ):
            net = html_re(str(tds[colm]))
            if net != '---':
                net = round(check_neg(str(tds), net) * (dollar_multiplier / 1_000_000))
                if r"this, 'defref_us-gaap_NetIncomeLossAvailableToCommonStockholdersBasic', window" in str(tds):
                    net_actual = True
        elif (r"this, 'defref_us-gaap_NetIncomeLossAttributableToNoncontrollingInterest', window" in str(tds) or
              r"this, 'defref_us-gaap_IncomeLossFromContinuingOperationsAttributableToNoncontrollingEntity', window" in str(tds) or
              r"this, 'defref_us-gaap_IncomeLossFromDiscontinuedOperationsNetOfTaxAttributableToNoncontrollingInterest', window" in str(tds)
              ):
            non_attributable_net = html_re(str(tds[colm]))
            if net != '---' and net_actual == False and non_attributable_net != '---':
                non_attributable_net = check_neg(str(tds), non_attributable_net)
                net -= round(non_attributable_net * (dollar_multiplier / 1_000_000))
        elif ('EarningsPerShareDiluted' in str(tds) and eps == '---' or
              'this, \'defref_us-gaap_EarningsPerShareBasicAndDiluted\', window' in str(tds) and eps == '---' or
              r"this, 'defref_us-gaap_IncomeLossFromContinuingOperationsPerBasicAndDilutedShare', window" in str(tds) and eps == '---'
              ):
            eps = html_re(str(tds[colm]))
            if eps != '---':
                eps = check_neg(str(tds), eps)
        elif (r"this, 'defref_us-gaap_CostOfRevenue', window" in str(tds) and cost == '---' or
              r"this, 'defref_us-gaap_CostOfGoodsSold', window" in str(tds) and cost == '---' or
              r"this, 'defref_us-gaap_CostOfGoodsAndServicesSold', window" in str(tds) and cost == '---' or
              r"this, 'defref_amgn_CostOfGoodsSoldExcludingAmortizationOfAcquiredIntangibleAssets', window" in str(tds) and cost == '---'
              ):
            result = html_re(str(tds[colm]))
            if result != '---':
                cost = round(check_neg(str(tds), result) * (dollar_multiplier / 1_000_000)) 
        elif (r"this, 'defref_us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding', window" in str(tds) or
              r"this, 'defref_us-gaap_WeightedAverageNumberOfShareOutstandingBasicAndDiluted', window" in str(tds) or
              r"this, 'defref_tsla_WeightedAverageNumberOfSharesOutstandingBasicAndDilutedOne', window" in str(tds)
              ):
            shares = html_re(str(tds[colm]))
            if shares != '---':
                share_set.add(round(shares * (share_multiplier / 1_000)))
        elif (r"this, 'defref_us-gaap_CommonStockDividendsPerShareDeclared', window" in str(tds) or
              r"this, 'defref_us-gaap_CommonStockDividendsPerShareCashPaid', window" in str(tds)
              ):
            div = html_re(str(tds[colm]))
        elif (r"this, 'defref_us-gaap_DepreciationAndAmortization', window" in str(tds) or
              r"this, 'defref_us-gaap_DepreciationDepletionAndAmortization', window" in str(tds)
              ):
            result = html_re(str(tds[colm]))
            if result != '---':
                dep_am = round(check_neg(str(tds), result) * (dollar_multiplier / 1_000_000)) 
        elif r"this, 'defref_us-gaap_AssetImpairmentCharges', window" in str(tds):
            result = html_re(str(tds[colm]))
            if result != '---':
                impairment = round(check_neg(str(tds), result) * (dollar_multiplier / 1_000_000)) 
        elif r"this, 'defref_us-gaap_GainsLossesOnSalesOfInvestmentRealEstate', window" in str(tds):
            disposition = html_re(str(tds[colm]))
            if disposition != '---':
                disposition = round(check_neg(str(tds), disposition) * (dollar_multiplier / 1_000_000))  
        elif r"this, 'defref_us-gaap_CostsAndExpenses', window" in str(tds):
            operating_exp_res = html_re(str(tds[colm]))
            if operating_exp_res != '---':
                operating_exp = round(check_neg(str(tds), operating_exp_res) * (dollar_multiplier / 1_000_000))         

    # Calculate share total
    share_sum = sum(share_set)

    # Calculate gross if not listed
    if gross == '---':
        try:
            gross = rev - cost
        except:
            gross = rev

    # Calculate shares if not listed
    if share_sum == 0 and net != '---' and eps != '---':
        share_sum = abs(round(net * 1000 / eps))

    # Calculate EPS if not listed
    if eps == '---' and net != '---' and share_sum != 0:
        eps = net / share_sum

    # Calculate FFO for REITS
    if net != '---':
        ffo = net + dep_am + impairment - disposition

    # Calculate OI if not listed
    if oi == '---' and operating_exp != 0:
        oi = rev - operating_exp

    return rev, gross, research, oi, net, eps, share_sum, div, ffo

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

    # Determine multiplier
    head = soup.find('th')
    dollar_multiplier = multiple_extractor(str(head))

    # Loop through rows, search for row of interest
    for row in soup.table.find_all('tr'):
        tds = row.find_all('td')
        if ('this, \'defref_us-gaap_CashAndCashEquivalentsAtCarryingValue\', window' in str(tds) or
            r"this, 'defref_us-gaap_CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents', window" in str(tds)
            ):
            cash_calc = html_re(str(tds[colm]))
            if cash_calc != '---':
                cash = round(check_neg(str(tds), cash_calc) * (dollar_multiplier / 1_000_000))
        elif ('defref_us-gaap_Goodwill' in str(tds) or
              r"this, 'defref_us-gaap_IntangibleAssetsNetIncludingGoodwill', window" in str(tds)
              ):
            goodwill_calc = html_re(str(tds[colm]))
            if goodwill_calc != '---':
                goodwill = round(check_neg(str(tds), goodwill_calc) * (dollar_multiplier / 1_000_000))
        elif ('this, \'defref_us-gaap_FiniteLivedIntangibleAssetsNet\', window' in str(tds) or
              'this, \'defref_us-gaap_IntangibleAssetsNetExcludingGoodwill\', window' in str(tds) or
              r"this, 'defref_us-gaap_IndefiniteLivedIntangibleAssetsExcludingGoodwill', window" in str(tds)
              ):
            intangible_assets_calc = html_re(str(tds[colm]))
            if intangible_assets_calc != '---':
                intangible_assets = (round(intangible_assets_calc * (dollar_multiplier / 1_000_000)))
                intangible_assets_set.add(intangible_assets)
        elif ('Total assets' in str(tds) or
              'Total Assets' in str(tds) or
              r"this, 'defref_us-gaap_Assets', window" in str(tds)
              ):
            assets_calc = html_re(str(tds[colm]))
            if assets_calc != '---':
                assets = round(check_neg(str(tds), assets_calc) * (dollar_multiplier / 1_000_000))
        elif 'onclick="top.Show.showAR( this, \'defref_us-gaap_Liabilities\', window )' in str(tds):
            liabilities_calc = html_re(str(tds[colm]))
            if liabilities_calc != '---':
                liabilities = round(check_neg(str(tds), liabilities_calc) * (dollar_multiplier / 1_000_000))
        elif ('this, \'defref_us-gaap_LongTermDebtNoncurrent\', window' in str(tds) or
              'this, \'defref_us-gaap_LongTermDebt\', window' in str(tds) or
              'this, \'defref_us-gaap_LongTermDebtAndCapitalLeaseObligations\', window' in str(tds) or
              r"this, 'defref_us-gaap_LineOfCredit', window" in str(tds) or
              r"this, 'defref_us-gaap_UnsecuredLongTermDebt', window" in str(tds) or
              r"this, 'defref_us-gaap_LongTermNotesPayable', window" in str(tds) or
              r"this, 'defref_stor_NonRecourseDebtNet', window" in str(tds) or
              r"this, 'defref_us-gaap_DebtInstrumentCarryingAmount', window" in str(tds)
              ):
            debt = html_re(str(tds[colm]))
            if debt == '---':
                debt = 0
            else:
                debt = round(debt * (dollar_multiplier / 1_000_000))
        elif 'this, \'defref_us-gaap_LiabilitiesAndStockholdersEquity\', window' in str(tds):
            tot_liabilities_calc = html_re(str(tds[colm]))
            if tot_liabilities_calc != '---':
                tot_liabilities = round(check_neg(str(tds), tot_liabilities_calc) * (dollar_multiplier / 1_000_000))
        elif ('this, \'defref_us-gaap_StockholdersEquity\', window' in str(tds) and equity =='---' or
              'this, \'defref_us-gaap_StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest\', window' in str(tds) and equity == '---'
              ):
            equity = html_re(str(tds[colm]))
            if equity != '---':
                equity = round(check_neg(str(tds), equity) * (dollar_multiplier / 1_000_000))          

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
    capex = share_issue = buyback = divpaid = sbc = 0
    cfo = fcf = debt_payment = '---'
    debt_pay = []
    capex_set = set()

    # Find which column has 12 month data
    colm = column_finder_annual_htm(soup)

    # Determine multiplier
    head = soup.find('th')
    dollar_multiplier = multiple_extractor(str(head))

    # Loop through rows, search for row of interest
    for row in soup.table.find_all('tr'):
        tds = row.find_all('td')
        if ('Net cash from operations' in str(tds) or
            'this, \'defref_us-gaap_NetCashProvidedByUsedInOperatingActivities\', window' in str(tds) or
            'this, \'defref_us-gaap_NetCashProvidedByUsedInOperatingActivitiesContinuingOperations\', window' in str(tds)
            ): 
            cfo_calc = html_re(str(tds[colm]))
            if cfo_calc != '---':
                cfo = round(check_neg(str(tds), cfo_calc) * (dollar_multiplier / 1_000_000))
        elif ('Additions to property and equipment' in str(tds) or
              'this, \'defref_us-gaap_PaymentsToAcquireProductiveAssets\', window' in str(tds) or
              'this, \'defref_us-gaap_PaymentsToAcquirePropertyPlantAndEquipment\', window' in str(tds) or
              'this, \'defref_us-gaap_PaymentsForProceedsFromProductiveAssets\', window' in str(tds) or
              r"this, 'defref_us-gaap_PaymentsToAcquireRealEstateHeldForInvestment', window" in str(tds) or
              r"this, 'defref_us-gaap_PaymentsToAcquireOtherPropertyPlantAndEquipment', window" in str(tds)
              ):
            capex_calc = html_re(str(tds[colm]))
            if capex_calc != '---':
                capex = (round(capex_calc * (dollar_multiplier / 1_000_000)))
                capex_set.add(capex)
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
                debt_pay.append(round(debt_payment * (dollar_multiplier / 1_000_000)))
        elif ('Common stock repurchased' in str(tds) or
              'this, \'defref_us-gaap_PaymentsForRepurchaseOfCommonStock\', window' in str(tds) or
              'Shares repurchased under stock compensation plans' in str(tds)
              ):
            buyback_calc = html_re(str(tds[colm]))
            if buyback_calc != '---':
                buyback += (round(buyback_calc * (dollar_multiplier / 1_000_000)))
        elif ('this, \'defref_us-gaap_ProceedsFromStockOptionsExercised\', window' in str(tds) or
              'this, \'defref_us-gaap_ProceedsFromIssuanceOfCommonStock\', window' in str(tds) or
              'this, \'defref_us-gaap_ProceedsFromIssuanceOfSharesUnderIncentiveAndShareBasedCompensationPlansIncludingStockOptions\', window' in str(tds)
              ):
            share_issue_rtn = html_re(str(tds[colm]))
            if share_issue_rtn != '---':
                share_issue += round(share_issue_rtn * (dollar_multiplier / 1_000_000))
        elif ('Common stock cash dividends paid' in str(tds) or
              'this, \'defref_us-gaap_PaymentsOfDividends\', window' in str(tds) or
              'PaymentsOfDividendsAndDividendEquivalentsOnCommonStockAndRestrictedStockUnits\', window' in str(tds) or
              'this, \'defref_us-gaap_PaymentsOfDividendsCommonStock\', window' in str(tds) or
              r"this, 'defref_us-gaap_Dividends', window" in str(tds) or
              r"this, 'defref_us-gaap_PaymentsOfOrdinaryDividends', window" in str(tds)
              ):
            divpaid_calc = html_re(str(tds[colm]))
            if divpaid_calc != '---':
                divpaid = (round(divpaid_calc * (dollar_multiplier / 1_000_000)))
        elif ('this, \'defref_us-gaap_ShareBasedCompensation\', window' in str(tds) or
              'this, \'defref_us-gaap_AllocatedShareBasedCompensationExpense\', window' in str(tds)
              ):
            sbc_calc = html_re(str(tds[colm]))
            if sbc_calc != '---':
                sbc = (round(sbc_calc * (dollar_multiplier / 1_000_000)))

    # Calculate Free Cash Flow
    fcf = cfo - sum(capex_set)

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
    if 'EQUITY' not in soup.find('th').text.upper() and 'Quarterly Financial Information' not in soup.find('th').text:
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

        if div == '---':
            obj = re.findall(r'(?:\$)(\d\.\d\d)(?:</font>)', str(soup))
            div = sum(list(map(float, obj[-4:])))
            return div

    elif 'Quarterly Financial Information' in soup.find('th').text:
        # Find column with total data        
        head = soup.table.find_all('tr')[5]
        tds = head.find_all('td')
        td_count = -1
        for td in tds:
            if 'colspan' in str(td):
                obj = re.search(r'(?:colspan=\")(\d)', str(td))
                td_count += int(obj.group(1))
            else:
                td_count += 1
            if 'Total' in str(td):
                break

        # Pull div data from table
        row_list = ['Dividends declared per common share', 'Cash dividends declared per common share']

        for title in row_list:
            try:
                table = pd.read_html(content, match=title)[1]
                row = table.loc[table[0] == title]
                div = float(row[td_count])
                break
            except:
                pass
        
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
            colm = column_finder_annual_htm(soup)

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
            pd.set_option('display.max_columns', None)
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
                
                elif 'Distributionper share' in table.values:
                    # For REITS
                    # Get bool dataframe with True at positions where the given value exists
                    result = table.isin(['Distributionper share'])
                    # Get list of columns that contains the value
                    series_obj = result.any()
                    column_number = list(series_obj[series_obj == True].index)
                    # Iterate over list of columns and fetch the rows indexes where value exists
                    for col in column_number:
                        row = list(result[col][result[col] == True].index)
                        row[0] += 1
                        answer = table.at[row[0], col]
                        try:
                            div = float(answer)
                            break
                        except:
                            continue
                    break
                
                elif 'Capital gains distribution' in table.values:
                    # For REITS
                    # Get bool dataframe with True at positions where the given value exists
                    result = table.isin(['Capital gains distribution'])
                    # Get list of columns that contains the value
                    series_obj = result.any()
                    column_number = list(series_obj[series_obj == True].index)
                    # Iterate over list of columns and fetch the rows indexes where value exists
                    for col in column_number:
                        row = list(result[col][result[col] == True].index)
                        answer = table.at[row[0] + 1, col + 2]
                        try:
                            div = float(answer)
                            break
                        except:
                            continue
                    break

                elif 'Return of capital' in table.values and 'Common Stock' in table.values:
                    # For REITS
                    # Get bool dataframe with True at positions where the given value exists
                    result = table.isin(['Return of capital'])
                    # Get list of columns that contains the value
                    series_obj = result.any()
                    column_number = list(series_obj[series_obj == True].index)
                    # Iterate over list of columns and fetch the rows indexes where value exists
                    for col in column_number:
                        row = list(result[col][result[col] == True].index)
                        answer = table.at[row[0] + 1, col + 2]
                        try:
                            div = float(answer)
                            break
                        except:
                            continue
                    break

    return div


def eps_catch_htm(catch_url, headers, eps):
    ''' Parse EPS and div info if not listed elsewhere

    Args:
        URL of statement
        User agent for SEC
    Returns:
        EPS (float)
        Dividend (float)
    '''

    # Initial value
    div = '---'

    # Get data from site
    content = requests.get(catch_url, headers=headers).content
    soup = BeautifulSoup(content, 'html.parser')

    # Find table
    table = pd.read_html(content)[1]

    # Pull eps data
    row = table.loc[table[0] == 'Diluted']
    for i in range(row.shape[1]):
        try:
            obj = re.search(r'(\d?\d\.\d\d)', str(row[i]))
            eps = float(obj.group(1).strip())
            break
        except:
            continue

    # Pull div data
    row = table.loc[table[0] == 'Class A']
    if row.empty:
        # Look for row with div data
        for index in range(table.shape[0]):
            if 'Less Dividends:' in str(table.iloc[index]):
                index += 1
                break
        
        # Pull div from row header
        search_str = table.iloc[index][0]
        try:
            obj = re.search(r'(?:\(\$)(\d?\d\.\d\d)(?:/share\))', search_str)
            div = float(obj.group(1).strip())
        except:
            div = '---'
    else:
        div = float(row[2])

    # Catch for div being randomly written on equity statement
    if 'Equity' in str(soup):
        if 'Board of Directors declared quarterly dividends per share of' in str(soup):
            obj = re.search(r'(?:Board of Directors declared quarterly dividends per share of)(?:.)*?(?:\$)(\d?\d.\d\d)', str(soup))
            div = float(obj.group(1).strip()) * 4
        elif 'the Board of Directors declared quarterly cash dividends of' in str(soup):
            obj = re.findall(r'(?:the Board of Directors declared quarterly cash dividends of)(?:.|\n)*?(\d?\d\.\d\d)', str(soup))
            div = float(obj[-1]) * 4
        elif 'the Board of Directors declared quarterly cash\n   dividends of' in str(soup):
            obj = re.search(r'(?:the Board of Directors declared quarterly cash\n\s{3}dividends of)(?:.)*?(?:\$)(\d?\d.\d\d)', str(soup))
            div = float(obj.group(1).strip()) * 4

    return eps, div


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
    share_sum = research = non_attributable_net = dep_am = impairment = disposition = ffo = 0
    net_check = False

    # Find which column has 12 month data
    colm = column_finder_annual_xml(soup)

    # Determine multiplier
    head = soup.RoundingOption
    dollar_multiplier, share_multiplier = multiple_extractor(str(head), shares=True, xml=True)

    # Loop through rows, search for row of interest
    rows = soup.find_all('Row')
    for row in rows:
        cells = row.find_all('Cell')
        if (r'<ElementName>us-gaap_Revenues</ElementName>' in str(row) or
            r'us-gaap_SalesRevenueNet' in str(row) or
            r'<ElementName>us-gaap_SalesRevenueGoodsNet</ElementName>' in str(row)
            ):
            rev = round(xml_re(str(cells[colm])) * (dollar_multiplier / 1_000_000))
        elif r'us-gaap_GrossProfit' in str(row):
            gross = xml_re(str(cells[colm]))
            if gross != '---':
                gross = round(check_neg(str(row), gross, 'xml')    * (dollar_multiplier / 1_000_000))     
        elif (r'us-gaap_CostOfRevenue' in str(row) and gross == '---' or
              r'us-gaap_CostOfGoodsAndServicesSold' in str(row) and gross == '---' or
              r'us-gaap_CostOfGoodsSold' in str(row) and gross == '---' or
              r'CostOfGoodsSoldExcludingAmortizationOfAcquiredIntangibleAssets' in str(row) and gross == '---'
              ):
            cost = round(xml_re(str(cells[colm])) * (dollar_multiplier / 1_000_000))             
        elif r'us-gaap_ResearchAndDevelopmentExpense' in str(row):
            research = round(xml_re(str(cells[colm])) * (dollar_multiplier / 1_000_000))                    
        elif r'us-gaap_OperatingIncomeLoss' in str(row):
            oi = xml_re(str(cells[colm]))
            if oi != '---':
                oi = round(check_neg(str(row), oi, 'xml') * (dollar_multiplier / 1_000_000))      
        elif (r'>us-gaap_NetIncomeLoss<' in str(row) and net_check == False or
              r'<ElementName>us-gaap_ProfitLoss</ElementName>' in str(row) and net == '---' and 'including non-controlling interest' not in str(row) and 'including noncontrolling interests' not in str(row)
              ):
            net = xml_re(str(cells[colm]))
            if net != '---':
                net = round(check_neg(str(row), net, 'xml') * (dollar_multiplier / 1_000_000))    
            if r'>us-gaap_NetIncomeLoss<' in str(row):
                net_check = True
        elif r'<ElementName>us-gaap_EarningsPerShareDiluted</ElementName>' in str(row) and eps == '---':
            eps = xml_re(str(cells[colm]))
            if eps != '---':
                eps = check_neg(str(row), eps, 'xml')
        elif r'us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding' in str(row):
            shares = round(xml_re(str(cells[colm])) * (share_multiplier / 1_000))
            share_sum += shares
        elif (r'us-gaap_CommonStockDividendsPerShareCashPaid' in str(row) or
              r'us-gaap_CommonStockDividendsPerShareDeclared' in str(row)):
            div = xml_re(str(cells[colm])) 
        elif r'<ElementName>us-gaap_DepreciationAmortizationAndAccretionNet</ElementName>' in str(row):
            dep_am = round(xml_re(str(cells[colm])) * (dollar_multiplier / 1_000_000))  
        elif r"this, 'defref_us-gaap_AssetImpairmentCharges', window" in str(row):              # Need to change
            impairment = round(xml_re(str(cells[colm])) * (dollar_multiplier / 1_000_000))    
        elif r"this, 'defref_us-gaap_GainsLossesOnSalesOfInvestmentRealEstate', window" in str(row):              # Need to change
            disposition = xml_re(str(cells[colm]))
            if disposition != '---':
                disposition = round(check_neg(str(row), oi, 'xml') * (dollar_multiplier / 1_000_000))  

    # Calculate gross if not listed
    if gross == '---':
        try:
            gross = rev - cost
        except:
            gross = rev

    # Calculate shares if not listed
    if share_sum == 0 and net != '---' and eps != '---':
        share_sum = abs(round(net * 1000 / eps))

    # Calculate EPS if not listed
    if eps == '---' and net != '---':
        eps = net / share_sum

    # Calculate FFO for REITS
    if net != '---':
        ffo = net + dep_am + impairment - disposition

    return rev, gross, research, oi, net, eps, share_sum, div, ffo

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
    
    # Determine multiplier
    head = soup.RoundingOption
    dollar_multiplier = multiple_extractor(str(head), xml=True)

    # Loop through rows, search for row of interest
    rows = soup.find_all('Row')
    for row in rows:
        cells = row.find_all('Cell')
        if ('us-gaap_CashCashEquivalentsAndShortTermInvestments' in str(row) or
            'us-gaap_CashAndCashEquivalentsAtCarryingValue' in str(row)
            ):
            cash = round(xml_re(str(cells[colm])) * (dollar_multiplier / 1_000_000)) 
        elif 'us-gaap_Goodwill' in str(row):
            goodwill = round(xml_re(str(cells[colm])) * (dollar_multiplier / 1_000_000))               
        elif r'us-gaap_IntangibleAssetsNetExcludingGoodwill' in str(row):
            intangible_assets = round(xml_re(str(cells[colm])) * (dollar_multiplier / 1_000_000))
        elif r'<ElementName>us-gaap_Assets</ElementName>' in str(row):
            assets = round(xml_re(str(cells[colm])) * (dollar_multiplier / 1_000_000))       
        elif (r'us-gaap_LongTermDebtNoncurrent' in str(row) or
              r'us-gaap_LongTermDebtAndCapitalLeaseObligations' in str(row)
              ):
            debt = xml_re(str(cells[colm]))
            if debt == '---':
                debt = 0
            else:
                debt = round(debt * (dollar_multiplier / 1_000_000))
        elif r'<ElementName>us-gaap_Liabilities</ElementName>' in str(row):
            liabilities = round(xml_re(str(cells[colm])) * (dollar_multiplier / 1_000_000))
        elif r'us-gaap_LiabilitiesAndStockholdersEquity' in str(row) and liabilities == '---':
            tot_liabilities = round(xml_re(str(cells[colm])) * (dollar_multiplier / 1_000_000))
        elif r'<ElementName>us-gaap_StockholdersEquity</ElementName>' in str(row):
            equity = xml_re(str(cells[colm])) 
            if equity != '---':
                equity = round(check_neg(str(row), equity, 'xml') * (dollar_multiplier / 1_000_000))                                                 

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
    capex = debt_pay = share_issue = buyback = divpaid = sbc = 0
    cfo = fcf = '---'

    # Find which column has 12 month data
    colm = column_finder_annual_xml(soup)

    # Determine multiplier
    head = soup.RoundingOption
    dollar_multiplier = multiple_extractor(str(head), xml=True)

    # Loop through rows, search for row of interest
    rows = soup.find_all('Row')
    for row in rows:
        cells = row.find_all('Cell')
        if (r'<ElementName>us-gaap_NetCashProvidedByUsedInOperatingActivities</ElementName>' in str(row) or
            r'<ElementName>us-gaap_NetCashProvidedByUsedInOperatingActivitiesContinuingOperations</ElementName>' in str(row)
            ):
            cfo = xml_re(str(cells[colm]))
            if cfo != '---':
                cfo = round(check_neg(str(row), cfo, 'xml') * (dollar_multiplier / 1_000_000)) 
        elif (r'us-gaap_PaymentsToAcquirePropertyPlantAndEquipment' in str(row) or
              r'us-gaap_PaymentsToAcquireProductiveAssets' in str(row)
              ):
            capex = round(xml_re(str(cells[colm])) * (dollar_multiplier / 1_000_000)) 
        elif (r'us-gaap_ProceedsFromRepaymentsOfShortTermDebtMaturingInThreeMonthsOrLess' in str(row) or
              r'RepaymentsOfShortTermAndLongTermBorrowings' in str(row) or
              r'us-gaap_RepaymentsOfLongTermDebtAndCapitalSecurities' in str(row) or
              r'us-gaap_InterestPaid' in str(row) or
              r'us-gaap_RepaymentsOfLongTermDebt' in str(row)
              ):
            debt_pay += round(xml_re(str(cells[colm])) * (dollar_multiplier / 1_000_000))                            
        elif (r'us-gaap_PaymentsForRepurchaseOfCommonStock' in str(row) or
              r'<ElementName>us-gaap_PaymentsForRepurchaseOfEquity</ElementName>' in str(row)
              ):
            buyback += round(xml_re(str(cells[colm])) * (dollar_multiplier / 1_000_000))    
        elif (r'us-gaap_ProceedsFromStockOptionsExercised' in str(row) or
              r'us-gaap_ProceedsFromIssuanceOfCommonStock' in str(row) or
              r'cost_ProceedsFromStockbasedAwardsNet' in str(row)
              ):
            share_issue = round(xml_re(str(cells[colm])) * (dollar_multiplier / 1_000_000))   
        elif (r'us-gaap_PaymentsOfDividendsCommonStock' in str(row) or
              r'us-gaap_PaymentsOfDividends' in str(row)
              ):
            divpaid = round(xml_re(str(cells[colm])) * (dollar_multiplier / 1_000_000))            
        elif r'us-gaap_ShareBasedCompensation' in str(row):
            sbc = round(xml_re(str(cells[colm])) * (dollar_multiplier / 1_000_000)) 
    
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

    # Get data from site
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


def eps_catch_xml(catch_url, headers, eps):
    ''' Parse EPS and div info if not listed elsewhere

    Args:
        URL of statement
        User agent for SEC
    Returns:
        EPS (float)
        Dividend (float)
    '''

    # Initial value
    div = '---'

    # Get data from site
    content = requests.get(catch_url, headers=headers).content
    soup = BeautifulSoup(content, 'xml')

    # Find table
    rows = soup.find_all('Row')
    for row in rows:
        if r'<ElementName>us-gaap_EarningsPerShareTextBlock</ElementName>' in str(row):
            cells = row.find_all('Cell')
            
            # Find eps
            obj = re.search(r'(?:Diluted)(?:.|\n)*?(?:\$)(\d?\d.\d\d)', str(cells[0]))
            if obj is not None:
                eps = float(obj.group(1))

            # Find div
            obj = re.search(r'(?:Less Dividends)(?:.|\n)*?(\d?\d.\d\d)', str(cells[0]))
            if obj is not None:
                div = float(obj.group(1))

    return eps, div