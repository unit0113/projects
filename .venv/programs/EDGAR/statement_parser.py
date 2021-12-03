from pandas.io.pytables import Table
import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime
import pandas as pd
import math
import numpy as np


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
        re_str = '.' + str(value)
    
    obj = re.search(re_str, string, re.M)
    
    # Check for weird float
    if obj == None:
        re_str = '.' + "{:,}".format(value, '.1f') + '.'
        obj = re.search(re_str, string, re.M)

    # Check if negative
    if type == 'htm':
        if '(' in obj.group(0) and ')' in obj.group(0):
            return (value * -1)
        else:
            return value
    else:
        if '-' in obj.group(0):
            return (value * -1)
        else:
            return value


# Find correct column for data
def column_finder_annual_htm(soup, per):
    ''' Determine which column of data contains desired data for htm pages

    Args:
        Soup (soup data of sub filings)
        Fiscal period end date
    Returns:
        Column number (int)        
    '''

    # Extract colmn header string
    head = soup.table.find_all('tr')[0]
    head2 = soup.table.find_all('tr')[1]
    first_cell = soup.table.find_all('tr')[2]

    # Identify next FY in case there is a more recent date than the relevant period
    next_year = str(int(per[-4:]) + 1)
    weird_colm = False

    # Find index for period end
    if per in str(head2):
        rows = head2.find_all('th')
        index = 0
        for row1 in rows:
            tweleve_month_prior = False
            # If extra/next FY in table
            if next_year in str(row1) or 'Minimum' in str(row1):
                if ('Subsequent Event [Member]' not in str(soup) and
                    'Shareholders\' Equity - Dividends Declared (Details)' not in str(head)
                    ):
                    weird_colm = True         
            elif per in str(row1):
                # Find correct column if multiple things before 12 months data
                col_index = 0
                colm = 0
                rows2 = head.find_all('th')

                # Go through colms and check if index lines up with 12 month data
                for row2 in rows2:
                    # Identify if 12 month data is listed prior and column is unlabled
                    if '12 Months Ended' in str(row2):
                        tweleve_month_prior = True
                    # Restart loop if data doesn't end on FY data, else, get new index and start loop over
                    if col_index > index:
                        if ('3 Months Ended' in str(row2) and ('12 Months Ended' in str(head) or '11 Months Ended' in str(head)) or
                            '4 Months Ended' in str(row2) or
                            '1 Months Ended' in str(row2) and '11 Months Ended' not in str(row2) or
                            '0 Months Ended' in str(row2) or
                            'Months Ended' not in str(row2) and '12 Months Ended' in str(head) and tweleve_month_prior == False
                            ):
                            break 
                        else:
                            # Check if data cells have wide cells
                            if 'colspan' in str(first_cell):
                                tds = first_cell.find_all('td')
                                col_index = 0
                                for cell_index, td in enumerate(tds):
                                    colm_part = re.search(r'(?:colspan=\"(\d\d?)\")', str(td), re.M)
                                    if colm_part == None:
                                        col_index += 1
                                    else:
                                        col_index += int(colm_part.group(1))
                                    if col_index > colm:
                                        return cell_index
                            else:
                                if weird_colm == True:
                                    return (colm + 1 - col_last)
                                else:
                                    return colm          
                    if 'colspan' in str(row2):
                        colm_part = re.search(r'(?:colspan=\"(\d\d?)\")', str(row2), re.M)
                        colm += int(colm_part.group(1))
                        col_index += int(colm_part.group(1))
                        col_last = int(colm_part.group(1))
                
            # Check for wide date columns
            if 'colspan' in str(row1):
                index_part = re.search(r'(?:colspan=\"(\d\d?)\")', str(row1), re.M)
                index += int(index_part.group(1))
            else:
                index += 1

        return (colm - col_last)

    else:
        # Find correct column accounting for empties prior to data
        colm = re.search(r'(?:colspan=\"(\d\d?)\")', str(head), re.M)
        return int(colm.group(1))


multiplier_list_1 = ['shares in Millions, $ in Millions', 'In Millions, except Per Share data, unless otherwise specified', 'In Millions, except Per Share data', 'In Millions', 'In Millions, unless otherwise specified']

multiplier_list_2 = ['shares in Thousands, $ in Millions', 'In Millions, except Share data in Thousands, unless otherwise specified']

multiplier_list_3 = ['shares in Thousands, $ in Thousands', 'In Thousands, except Per Share data, unless otherwise specified', 'In Thousands, except Per Share data', 'In Thousands, unless otherwise specified', 'In Thousands', '$ / shares in Units, shares in Thousands, $ in Thousands',
                     '$ / shares in Units, $ in Thousands']

multiplier_list_4 = ['In Thousands, except Share data, unless otherwise specified', 'In Thousands, except Share data']

multiplier_list_5 = ['$ in Millions', '$ in Millions, ¥ in Billions', 'In Millions, except Share data', 'In Millions, except Share data, unless otherwise specified', '€ in Millions, $ in Millions']

multiplier_list_6 = ['$ in Thousands']

multiplier_list_7 = ['shares in Thousands']

multiplier_list_8 = ['shares in Millions']

multiplier_list_9 = ['Share data in Thousands, except Per Share data, unless otherwise specified']


def multiple_extractor(head, shares=False, xml=False):
    ''' Return $ and share multiplier for statement

    Args:
        Head of statement
        Bool for whether to return share multiplier or not
        Bool for whether this is an xml doc or not
    Returns:
        Dollar multiplier (int)
        Share multiplier (int) (optional)     
    ''' 

    # Search for multiplier string
    if xml == True:
        obj = re.search(r'(?:<RoundingOption>)(.*)(?:</RoundingOption>)', head)
    else: 
        obj = re.search(r'(?:<br/?>)(.*)(?:</strong>)', head)
    if obj != None:
        result = obj.group(1).strip()
    else:
        if shares == True:
            return 1, 1
        else:
            return 1

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
            return 1_000_000, 1
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
    elif result in multiplier_list_9 and shares == True:
        return 1, 1_000
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
        Fiscal period end date
        Name of company      
    '''

    # Get data from site
    content = requests.get(sum_url, headers=headers).content
    soup = BeautifulSoup(content, 'html.parser')

    # Initial values
    fy = period_end = name = '---'
    colm = 1

    # Loop through rows, search for FY and period end date
    for row in soup.table.find_all('tr'):
        tds = row.find_all('td')
        if r"this, 'defref_dei_DocumentFiscalYearFocus', window" in str(tds):
            try:
                fy = int(tds[colm].text)
            except:
                # Account for weird extra column in front of data
                colm += 1
                fy = int(tds[colm].text)
        elif r"this, 'defref_dei_DocumentPeriodEndDate', window" in str(tds):
            period_end_str = tds[colm].text.strip()
            period_end = period_end_str.replace('  ', ' ')
            if period_end == '':
                colm += 1
                period_end_str = tds[colm].text.strip()
                period_end = period_end_str.replace('  ', ' ')
        elif r"this, 'defref_dei_EntityRegistrantName', window" in str(tds):
            name = str(tds[colm].text.strip())
            if name == '':
                colm += 1
                name = str(tds[colm].text.strip())
            name = name.replace(' /DE/', '')
        elif '[Member]' in str(row) and fy != '---':
            break 

    # Remove extra spaces and line breaks from date
    period_end = " ".join(period_end.split())

    return fy, period_end, name


'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


def rev_htm(rev_url, headers, per):    
    ''' Parse .htm income statement

    Args:
        URL of statement
        User agent for SEC
        Fiscal period end date
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
    gross = oi = net = eps = shares = div = '---'
    credit_loss_provision = cost_sum = cost = rev = op_exp = share_sum = research = non_attributable_net = dep_am = impairment = disposition = ffo = operating_exp = int_rev = oth_rev = 0
    share_set = set()
    net_actual = net_add = False

    # Find which column has 12 month data
    colm = column_finder_annual_htm(soup, per)

    # Determine multiplier
    head = soup.find('th')
    dollar_multiplier, share_multiplier = multiple_extractor(str(head), shares=True)

    # Loop through rows, search for row of interest
    for row in soup.table.find_all('tr'):
        tds = row.find_all('td')
        if tds == []:
            continue

        elif (r"this, 'defref_us-gaap_Revenues', window" in str(tds[0]) or
            r"this, 'defref_us-gaap_SalesRevenueNet', window" in str(tds[0]) or
            r'defref_us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax' in str(tds[0]) or
            r"this, 'defref_us-gaap_SalesRevenueGoodsNet', window" in str(tds[0]) or
            r"this, 'defref_us-gaap_RealEstateRevenueNet', window" in str(tds[0]) or
            r"this, 'defref_us-gaap_RevenueFromContractWithCustomerIncludingAssessedTax', window" in str(tds[0]) or
            r"this, 'defref_us-gaap_RegulatedAndUnregulatedOperatingRevenue', window" in str(tds[0]) or
            r"this, 'defref_us-gaap_SalesRevenueServicesNet', window" in str(tds[0]) or
            r"this, 'defref_us-gaap_ElectricUtilityRevenue', window" in str(tds[0]) or
            r"this, 'defref_us-gaap_UtilityRevenue', window" in str(tds[0]) or
            r"this, 'defref_us-gaap_RevenuesNetOfInterestExpense', window" in str(tds[0]) or
            r"this, 'defref_gs_NetRevenuesIncludingNetInterestIncome', window" in str(tds[0]) or
            r"this, 'defref_us-gaap_RevenueMineralSales', window" in str(tds[0])
            ):
            rev_calc = html_re(str(tds[colm]))
            if rev_calc == '---':
                rev_calc = html_re(str(tds[1]))
                if rev_calc != '---':
                    colm = 1
            if rev_calc != '---':
                rev_calc = round(rev_calc * (dollar_multiplier / 1_000_000), 2)
                if rev_calc > rev:
                    rev = rev_calc

        elif (r"this, 'defref_us-gaap_InterestIncomeOperating', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_InterestAndDividendIncomeOperating', window" in str(tds[0])
              ):
            int_rev_calc = html_re(str(tds[colm]))
            if int_rev_calc != '---':
                int_rev = round(int_rev_calc * (dollar_multiplier / 1_000_000), 2)

        elif (r"this, 'defref_us-gaap_NoninterestIncome', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_OtherIncome', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_ContractuallySpecifiedServicingFeesLateFeesAndAncillaryFeesEarnedInExchangeForServicingFinancialAssets', window" in str(tds[0]) or
              r"this, 'defref_nrz_GainLossOnSaleOfOriginatedMortgageLoansHeldForSaleNet', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_NonoperatingGainsLosses', window" in str(tds[0])
              ):
            oth_rev_calc = html_re(str(tds[colm]))
            if oth_rev_calc != '---':
                oth_rev += round(check_neg(str(tds[colm]), oth_rev_calc) * (dollar_multiplier / 1_000_000), 2)

        elif r'defref_us-gaap_GrossProfit' in str(tds[0]):
            gross = html_re(str(tds[colm]))
            if gross != '---':
                gross = round(check_neg(str(tds[colm]), gross) * (dollar_multiplier / 1_000_000), 2)

        elif (r"this, 'defref_us-gaap_ResearchAndDevelopmentExpense', window" in str(tds[0]) or
              r"this, 'defref_amzn_TechnologyAndContentExpense', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost', window" in str(tds[0])
              ):
            research_calc = html_re(str(tds[colm]))
            if research_calc != '---':
                research = round(research_calc * (dollar_multiplier / 1_000_000), 2)

        elif (r"this, 'defref_us-gaap_CostsAndExpenses', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_OperatingExpenses', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_NoninterestExpense', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_OperatingCostsAndExpenses', window" in str(tds[0]) or
              r"this, 'defref_o_ExpensesIncludingIncomeTaxExpense', window" in str(tds[0])
              ):
            result = html_re(str(tds[colm]))
            if result != '---':
                operating_exp = round(check_neg(str(tds[colm]), result) * (dollar_multiplier / 1_000_000), 2)

        elif (r"this, 'defref_us-gaap_ProvisionForLoanLeaseAndOtherLosses', window" in str(tds[0])
              ):
            result = html_re(str(tds[colm]))
            if result != '---':
                credit_loss_provision = round(check_neg(str(tds[colm]), result) * (dollar_multiplier / 1_000_000), 2)     

        elif (r"this, 'defref_us-gaap_SellingGeneralAndAdministrativeExpense', window" in str(tds[0]) and oi == '---' or
              r"this, 'defref_us-gaap_AdvertisingExpense', window" in str(tds[0]) and oi == '---' or
              r"this, 'defref_us-gaap_AssetImpairmentCharges', window" in str(tds[0]) and oi == '---' or
              r"this, 'defref_us-gaap_GeneralAndAdministrativeExpense', window" in str(tds[0]) and oi == '---' or
              r"this, 'defref_abr_PropertyOperatingExpense', window" in str(tds[0]) and oi == '---' or
              r"this, 'defref_us-gaap_LaborAndRelatedExpense', window" in str(tds[0]) and oi == '---' or
              r"this, 'defref_tmo_RestructuringAndOtherCostsIncomeNet', window" in str(tds[0]) and oi == '---'
              ):
            op_exp_calc = html_re(str(tds[colm]))
            if op_exp_calc != '---':
                op_exp += round(op_exp_calc * (dollar_multiplier / 1_000_000), 2)

        elif (r"this, 'defref_us-gaap_OperatingIncomeLoss', window" in str(tds[0]) and oi == '---' or
              r"this, 'defref_pru_IncomeLossFromContinuingOperationsBeforeIncomeTaxesAndOperatingJointVentures', window" in str(tds[0]) and oi == '---'
              ):
            oi_calc = html_re(str(tds[colm]))
            if oi_calc != '---':
                oi = round(check_neg(str(tds[colm]), oi_calc) * (dollar_multiplier / 1_000_000), 2)

        elif ('this, \'defref_us-gaap_NetIncomeLoss\', window' in str(tds[0]) and net == '---' or
              'this, \'defref_us-gaap_NetIncomeLoss\', window' in str(tds[0]) and net_actual == False or
              'this, \'defref_us-gaap_ProfitLoss\', window' in str(tds[0]) and net == '---' or
              r"this, 'defref_us-gaap_NetIncomeLossAvailableToCommonStockholdersBasic', window" in str(tds[0]) and net_actual == False or
              r"this, 'defref_us-gaap_IncomeLossFromContinuingOperations', window" in str(tds[0]) and net == '---' or
              r"this, 'defref_us-gaap_IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest', window" in str(tds[0]) and net == '---' or
              r"this, 'defref_us-gaap_NetIncomeLossAvailableToCommonStockholdersBasic', window" in str(tds[0]) and ('>Net earnings attributable to common stockholders<' in str(tds[0]) or 'Net income (loss) attributable to common stockholders' in str(tds[0]) or '>Net earnings available to common stockholders<' in str(tds[0]))
            ):
            net_calc = html_re(str(tds[colm]))
            if net_calc != '---':
                net = round(check_neg(str(tds[colm]), net_calc) * (dollar_multiplier / 1_000_000), 2)
                if (r"this, 'defref_us-gaap_NetIncomeLossAvailableToCommonStockholdersBasic', window" in str(tds[0]) or
                    'ATTRIBUT' in str(tds[0]).upper() and 'NONCONTROLLING' not in str(tds[0]).upper() and 'NON-CONTROLLING' not in str(tds[0]).upper()
                    ):
                    net_actual = True

        elif (r"this, 'defref_us-gaap_NetIncomeLossAttributableToNoncontrollingInterest', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_IncomeLossFromContinuingOperationsAttributableToNoncontrollingEntity', window" in str(tds[0]) and non_attributable_net == 0 or
              r"this, 'defref_us-gaap_IncomeLossFromDiscontinuedOperationsNetOfTaxAttributableToNoncontrollingInterest', window" in str(tds[0])  and non_attributable_net == 0
              ):
            non_attributable_net_calc = html_re(str(tds[colm]))
            if non_attributable_net_calc != '---':
                non_attributable_net = round(check_neg(str(tds[colm]), non_attributable_net_calc) * (dollar_multiplier / 1_000_000), 2)
                if ('Net income attributable to noncontrolling interests' in str(tds[0]) or
                    'Net earnings from continuing operations attributable to noncontrolling interests' in str(tds[0]) or
                    'Net loss from discontinued operations attributable to noncontrolling interests' in str(tds[0]) or
                    'Net income attributable to non-controlling interests' in str(tds[0]) or
                    'Net (earnings) losses attributable to noncontrolling interests' in str(tds[0]) or
                    'NET LOSS ATTRIBUTABLE TO NONCONTROLLING INTERESTS' in str(tds[0]) or
                    'Noncontrolling interests, net of income taxes' in str(tds[0]) or
                    'Net (income) loss attributable to noncontrolling interests' in str(tds[0]) or
                    'Total net income attributable to noncontrolling interests' in str(tds[0]) or
                    'Net Income Attributable to Noncontrolling Interest' in str(tds[0])
                    ):
                    net_add = True

        elif (r"this, 'defref_us-gaap_EarningsPerShareDiluted', window" in str(tds[0]) and eps == '---' or
              'this, \'defref_us-gaap_EarningsPerShareBasicAndDiluted\', window' in str(tds[0]) and eps == '---' or
              r"this, 'defref_us-gaap_IncomeLossFromContinuingOperationsPerBasicAndDilutedShare', window" in str(tds[0]) and eps == '---' or
              r"this, 'defref_us-gaap_IncomeLossFromContinuingOperationsPerDilutedShare', window" in str(tds[0]) and eps == '---' and '>Continuing operations' not in str(tds[0]) and r"this, 'defref_us-gaap_EarningsPerShareDiluted', window" not in str(soup) or
              r"this, 'defref_o_Diluted', window" in str(tds[0])
              ):
            result = html_re(str(tds[colm]))
            if result != '---' and result != 0:
                eps = check_neg(str(tds[colm]), result)

        elif (r"this, 'defref_us-gaap_CostOfRevenue', window" in str(tds[0]) and net == '---' or
              r"this, 'defref_us-gaap_CostOfGoodsSold', window" in str(tds[0]) and cost == 0 or
              r"this, 'defref_us-gaap_CostOfGoodsAndServicesSold', window" in str(tds[0]) and cost == 0 or
              r"this, 'defref_amgn_CostOfGoodsSoldExcludingAmortizationOfAcquiredIntangibleAssets', window" in str(tds[0]) and cost == 0 or
              r"this, 'defref_nee_FuelPurchasedPowerAndInterchangeExpense', window" in str(tds[0]) and cost == 0 or
              r"this, 'defref_us-gaap_CostOfGoodsAndServicesEnergyCommoditiesAndServices', window" in str(tds[0]) and cost == 0 or
              r"this, 'defref_us-gaap_MineralExtractionProcessingAndMarketingCosts', window" in str(tds[0]) and cost == 0
              ):
            result = html_re(str(tds[colm]))
            if result != '---':
                cost = round(result * (dollar_multiplier / 1_000_000), 2) 

        elif (r"this, 'defref_us-gaap_InterestExpense', window" in str(tds[0]) and cost == 0 and oi == '---' or
              r"this, 'defref_abr_PropertyOperatingExpense', window" in str(tds[0]) and cost == 0 and oi == '---' or
              r"this, 'defref_stor_PropertyCosts', window" in str(tds[0]) and cost == 0 and oi == '---' or
              r"this, 'defref_us-gaap_UtilitiesOperatingExpenseProductsAndServices', window" in str(tds[0]) and cost == 0 and oi == '---' or
              r"this, 'defref_dte_UnregulatedOperatingExpenseProductsandServices', window" in str(tds[0]) and cost == 0 and oi == '---' or
              r"this, 'defref_dte_UtilitiesOperatingExpenseFuelUsedPurchasedPowerandGasandPetroleumPurchased', window" in str(tds[0]) and cost == 0 and oi == '---' or
              r"this, 'defref_dte_UnregulatedOperatingExpenseFuelUsedPurchasedPowerandGasandPetroleumPurchased', window" in str(tds[0]) and cost == 0 and oi == '---' or
              r"this, 'defref_dte_RegulatedOperatingExpenseFuelUsedPurchasedPowerGas', window" in str(tds[0]) and cost == 0 and oi == '---' or
              r"this, 'defref_dte_UnregulatedOperatingExpenseFuelUsedPurchasedPowerGas', window" in str(tds[0]) and cost == 0 and oi == '---' or
              r"this, 'defref_us-gaap_CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization', window" in str(tds[0]) and cost == 0 and oi == '---' or
              r"this, 'defref_us-gaap_PolicyholderBenefitsAndClaimsIncurredNet', window" in str(tds[0]) and cost == 0 and oi == '---' or
              r"this, 'defref_us-gaap_InterestCreditedToPolicyholdersAccountBalances', window" in str(tds[0]) and cost == 0 and oi == '---' or
              r"this, 'defref_us-gaap_PolicyholderDividends', window" in str(tds[0]) and cost == 0 and oi == '---' or
              r"this, 'defref_us-gaap_DeferredPolicyAcquisitionCostAmortizationExpense', window" in str(tds[0]) and cost == 0 and oi == '---'
              ):
            result = html_re(str(tds[colm]))
            if result != '---':
                cost_sum += round(result * (dollar_multiplier / 1_000_000), 2)

        elif (r"this, 'defref_us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_WeightedAverageNumberOfShareOutstandingBasicAndDiluted', window" in str(tds[0]) or
              r"this, 'defref_tsla_WeightedAverageNumberOfSharesOutstandingBasicAndDilutedOne', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_WeightedAverageNumberDilutedSharesOutstandingAdjustment', window" in str(tds[0])
              ):
            shares = html_re(str(tds[colm]))
            if shares != '---':
                share_set.add(round(shares * (share_multiplier / 1_000), 2))

        elif (r"this, 'defref_us-gaap_CommonStockDividendsPerShareDeclared', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_CommonStockDividendsPerShareCashPaid', window" in str(tds[0])
              ):
            div = html_re(str(tds[colm]))

        elif (r"this, 'defref_us-gaap_DepreciationAndAmortization', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_DepreciationDepletionAndAmortization', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_Depreciation', window" in str(tds[0])
              ):
            result = html_re(str(tds[colm]))
            if result != '---':
                dep_am = round(check_neg(str(tds[colm]), result) * (dollar_multiplier / 1_000_000), 2) 

        elif (r"this, 'defref_us-gaap_AssetImpairmentCharges', window" in str(tds[0]) or
              r"this, 'defref_nnn_ImpairmentLossesAndOtherChargesNetOfRecoveries', window" in str(tds[0]) or
              r"this, 'defref_nrz_TotalImpairmentCharges', window" in str(tds[0]) or
              r"this, 'defref_nrz_ImpairmentNetOfTheReversalOfPriorValuationAllowancesOnLoans', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_OtherThanTemporaryImpairmentLossesInvestmentsAvailableforsaleSecurities', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_ProvisionForLoanLeaseAndOtherLosses', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_ImpairmentOfRealEstate', window" in str(tds[0])
              ):
            result = html_re(str(tds[colm]))
            if result != '---':
                impairment += round(check_neg(str(tds[colm]), result) * (dollar_multiplier / 1_000_000), 2) 

        elif r"this, 'defref_us-gaap_GainsLossesOnSalesOfInvestmentRealEstate', window" in str(tds[0]):
            result = html_re(str(tds[colm]))
            if result != '---':
                disposition = round(check_neg(str(tds[colm]), result) * (dollar_multiplier / 1_000_000), 2)  
      
        elif ('[Member]' in str(row) and rev != 0 and 'Financial Services Business [Member]' not in str(row) or
              'Successor' in str(row) and rev != 0
              ):
            break   

    # Check for gross data in member section
    if gross == '---' and r"this, 'defref_us-gaap_CostOfGoodsAndServicesSold', window" in str(soup) and cost == '---':
        cost = 0
        for row in soup.table.find_all('tr'):
            tds = row.find_all('td')
            if tds == []:
                continue
            elif (r"this, 'defref_us-gaap_CostOfRevenue', window" in str(tds[0]) or
                r"this, 'defref_us-gaap_CostOfGoodsSold', window" in str(tds[0]) or
                r"this, 'defref_us-gaap_CostOfGoodsAndServicesSold', window" in str(tds[0]) or
                r"this, 'defref_amgn_CostOfGoodsSoldExcludingAmortizationOfAcquiredIntangibleAssets', window" in str(tds[0])
                ):
                result = html_re(str(tds[colm]))
                if result != '---':
                    cost += round(result * (dollar_multiplier / 1_000_000), 2) 

    # Check for div data in member section
    if div == '---' and r"this, 'defref_us-gaap_CommonStockDividendsPerShareDeclared', window" in str(soup):
        for row in soup.table.find_all('tr'):
            tds = row.find_all('td')
            if tds == []:
                continue
            elif (r"this, 'defref_us-gaap_CommonStockDividendsPerShareDeclared', window" in str(tds[0])
                ):
                div = html_re(str(tds[colm]))

    # Calculate rev for REITs if total not given
    if rev < int_rev + oth_rev and 'Other income' in str(soup) and int_rev != 0 and oth_rev != 0:
        rev = round(int_rev + oth_rev, 2)

    # Find highest value of cost and cost_sum
    cost = max(cost, cost_sum)

    # Calculate operating income if not found
    op_exp += cost + research + dep_am
    if oi == '---' and operating_exp > 0:
        oi = round(rev - credit_loss_provision - operating_exp, 2)
    elif oi == '---' and op_exp > 0:
        oi = round(rev - credit_loss_provision - op_exp, 2)

    # Calculate Gross if OI is given
    if cost == 0 and oi != '---' and operating_exp != 0:
        cost = operating_exp - op_exp
        gross = round(rev - cost, 2)

    # Calculate share total
    share_sum = round(sum(share_set), 2)
    if share_sum == 0:
        share_sum = '---'

    # Account for non_attributable income
    if net != '---' and net_actual == False and non_attributable_net != '---':
        if net_add == True:
            net = round(net + non_attributable_net, 2)
        else:
            net = round(net - non_attributable_net, 2)

    # Calculate gross if not listed
    if gross == '---':
        if cost != 0:
            gross = round(rev - cost, 2)
        else:
            gross = rev

    # Calculate FFO for REITS
    if net != '---':
        ffo = round(net + dep_am + impairment - disposition, 2)

    return rev, gross, research, oi, net, eps, share_sum, div, ffo

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

def bs_htm(bs_url, headers, per):    
    ''' Parse .htm balance sheet statement

    Args:
        URL of statement
        User agent for SEC
        Fiscal period end date
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
    equity = cash = cur_assets = assets = cur_liabilities = liabilities = tot_liabilities = '---'
    cash_sum = intangible_assets = goodwill = recievables = debt = cur_liabilities_sum = 0
    intangible_assets_set = set()

    # Find which column has 12 month data
    colm = column_finder_annual_htm(soup, per)

    # Determine multiplier
    head = soup.find('th')
    dollar_multiplier = multiple_extractor(str(head))
    
    # Loop through rows, search for row of interest
    for row in soup.table.find_all('tr'):
        tds = row.find_all('td')
        if tds == []:
            continue
        
        elif ('this, \'defref_us-gaap_CashAndCashEquivalentsAtCarryingValue\', window' in str(tds[0]) or
            r"this, 'defref_us-gaap_CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents', window" in str(tds[0]) or
            r"this, 'defref_us-gaap_CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsIncludingDisposalGroupAndDiscontinuedOperations', window" in str(tds[0]) or
            r"this, 'defref_us-gaap_CashEquivalentsAtCarryingValue', window" in str(tds[0]) or
            r"this, 'defref_us-gaap_Cash', window" in str(tds[0]) or
            r"this, 'defref_us-gaap_CashCashEquivalentsAndShortTermInvestments', window" in str(tds[0]) or
            r"this, 'defref_gs_CashAndCashEquivalents', window" in str(tds[0])
            ):
            cash_calc = html_re(str(tds[colm]))
            if cash_calc != '---':
                cash = round(check_neg(str(tds[colm]), cash_calc) * (dollar_multiplier / 1_000_000), 2)

        elif (r"this, 'defref_us-gaap_CashAndDueFromBanks', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_InterestBearingDepositsInBanks', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_RestrictedCash', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_CashReserveDepositRequiredAndMade', window" in str(tds[0])
              ):
            cash_calc = html_re(str(tds[colm]))
            if cash_calc != '---':
                cash_sum += round(check_neg(str(tds[colm]), cash_calc) * (dollar_multiplier / 1_000_000), 2)

        elif ('defref_us-gaap_Goodwill' in str(tds[0]) or
              r"this, 'defref_us-gaap_IntangibleAssetsNetIncludingGoodwill', window" in str(tds[0])
              ):
            goodwill_calc = html_re(str(tds[colm]))
            if goodwill_calc != '---':
                goodwill = round(check_neg(str(tds[colm]), goodwill_calc) * (dollar_multiplier / 1_000_000), 2)

        elif ('this, \'defref_us-gaap_FiniteLivedIntangibleAssetsNet\', window' in str(tds[0]) or
              'this, \'defref_us-gaap_IntangibleAssetsNetExcludingGoodwill\', window' in str(tds[0]) or
              r"this, 'defref_us-gaap_IndefiniteLivedIntangibleAssetsExcludingGoodwill', window" in str(tds[0])
              ):
            intangible_assets_calc = html_re(str(tds[colm]))
            if intangible_assets_calc != '---':
                intangible_assets = round(intangible_assets_calc * (dollar_multiplier / 1_000_000), 2)
                intangible_assets_set.add(intangible_assets)

        elif ('Total assets' in str(tds[0]) and assets == '---' or
              'Total Assets' in str(tds[0]) and assets == '---' or
              r"this, 'defref_us-gaap_Assets', window" in str(tds[0]) and assets == '---'
              ):
            assets_calc = html_re(str(tds[colm]))
            if assets_calc != '---':
                assets = round(check_neg(str(tds[colm]), assets_calc) * (dollar_multiplier / 1_000_000), 2)

        elif (r"this, 'defref_us-gaap_AssetsCurrent', window" in str(tds[0])
              ):
            cur_assets_calc = html_re(str(tds[colm]))
            if cur_assets_calc != '---':
                cur_assets = round(check_neg(str(tds[colm]), cur_assets_calc) * (dollar_multiplier / 1_000_000), 2)

        elif (r"this, 'defref_us-gaap_LoansAndLeasesReceivableNetReportedAmount', window" in str(tds[0]) or
              r"this, 'defref_mpw_InterestAndRentReceivable', window" in str(tds[0]) or
              r"this, 'defref_mpw_StraightLineRentReceivable', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_DueFromRelatedParties', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_FederalFundsSoldAndSecuritiesPurchasedUnderAgreementsToResell', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_SecuritiesBorrowed', window" in str(tds[0]) or
              r"this, 'defref_gs_CustomerAndOtherReceivables', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_LoansAndLeasesReceivableNetReportedAmount', window" in str(tds[0]) or
              r"this, 'defref_gs_CashAndSecuritiesSegregatedForRegulatoryAndOtherPurposes', window" in str(tds[0]) or
              r"this, 'defref_gs_SecuritiesPurchasedUnderAgreementsToResellAndFederalFundsSold', window" in str(tds[0]) or
              r"this, 'defref_gs_ReceivablesFromBrokersDealersAndClearingOrganizationsBS', window" in str(tds[0]) or
              r"this, 'defref_gs_ReceivablesFromCustomersAndCounterparties', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_ReceivablesFromBrokersDealersAndClearingOrganizations', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_ReceivablesFromCustomers', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_SecuritiesPurchasedUnderAgreementsToResell', window" in str(tds) or
              r"this, 'defref_us-gaap_OtherReceivables', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_NotesReceivableNet', window" in str(tds[0]) and 'Commercial mortgage and other loans' not in str(tds[0]) or
              r"this, 'defref_us-gaap_LoansReceivableHeldForSaleNetNotPartOfDisposalGroup', window" in str(tds[0]) or
              r"this, 'defref_ms_LoansHeldforInvestment', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_SecuritiesReceivedAsCollateral', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_LoansReceivableHeldForSaleNet', window" in str(tds[0]) or
              r"this, 'defref_ms_Receivables', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_AccountsReceivableNet', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_AccruedFeesAndOtherRevenueReceivable', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_NotesAndLoansReceivableNetNoncurrent', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_AccountsAndNotesReceivableNet', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_DeferredRentReceivablesNet', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_InterestsContinuedToBeHeldByTransferorFairValue', window" in str(tds[0]) or
              r"this, 'defref_nrz_ExcessMortgageServicingRightsFairValue', window" in str(tds[0]) or
              r"this, 'defref_nrz_MortgageServicingRightsFairValue', window" in str(tds[0]) or
              r"this, 'defref_nrz_MortgageServicingRightsFinancingReceivableatFairValue', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_RestrictedCashAndCashEquivalents', window" in str(tds[0]) or
              r"this, 'defref_nrz_ResidentialMortgageLoansHeldforinvestment', window" in str(tds[0]) or
              r"this, 'defref_nrz_ResidentialMortgageLoansSubjectToRepurchase', window" in str(tds[0]) or
              r"this, 'defref_nrz_ServicerAdvancesReceivableNet', window" in str(tds[0]) or
              r"o_AccountsReceivable" in str(tds[0]) or
              r"this, 'defref_us-gaap_ContractWithCustomerAssetNet', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_MortgageLoansOnRealEstateCommercialAndConsumerNet', window" in str(tds[0])
              ):
            if cur_assets == '---':
                recievables_calc = html_re(str(tds[colm]))
                if recievables_calc != '---':
                    recievables += round(check_neg(str(tds[colm]), recievables_calc) * (dollar_multiplier / 1_000_000), 2)

        elif r"this, 'defref_us-gaap_LiabilitiesCurrent', window" in str(tds[0]) and cur_liabilities == '---':
            cur_liabilities_calc = html_re(str(tds[colm]))
            if cur_liabilities_calc != '---':
                cur_liabilities = round(check_neg(str(tds[colm]), cur_liabilities_calc) * (dollar_multiplier / 1_000_000), 2)

        elif (r"this, 'defref_us-gaap_UnsecuredDebt', window" in str(tds[0]) and 'nrz' not in str(soup) or
              r"this, 'defref_stor_AccruedExpensesDeferredRevenueAndOtherLiabilities', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_AccountsPayableAndAccruedLiabilitiesCurrentAndNoncurrent', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_ConvertibleNotesPayable', window" in str(tds[0]) and 'Accounted for using the operating method, net of accumulated depreciation and amortization' not in str(soup) or
              r"this, 'defref_us-gaap_AccountsPayableRelatedPartiesCurrentAndNoncurrent', window" in str(tds[0]) or
              r"this, 'defref_abr_DueToBorrowers', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_AccountsPayableCurrentAndNoncurrent', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_AccruedLiabilitiesAndOtherLiabilities', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_DerivativeLiabilities', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_Deposits', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_SecuritiesSoldUnderAgreementsToRepurchase', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_SecuritiesLoaned', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_OtherSecuredFinancings', window" in str(tds[0]) or
              r"this, 'defref_gs_CustomerAndOtherPayables', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_TradingLiabilities', window" in str(tds[0]) or
              r"this, 'defref_gs_UnsecuredShortTermBorrowingsIncludingCurrentPortionOfUnsecuredLongTermBorrowings', window" in str(tds[0]) or
              r"this, 'defref_gs_SecuritiesLoanedBS', window" in str(tds[0]) or
              r"this, 'defref_gs_PayablesToBrokerDealersAndClearingOrganizationsBS', window" in str(tds[0]) or
              r"this, 'defref_gs_PayablesToCustomersAndCounterparties', window" in str(tds[0]) or
              r"this, 'defref_gs_UnsecuredShortTermBorrowingsIncludingCurrentPortionOfUnsecuredLongTermBorrowings', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_FinancialInstrumentsSoldNotYetPurchasedAtFairValue', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_PayablesToBrokerDealersAndClearingOrganizations', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_PayablesToCustomers', window" in str(tds[0]) or
              r"this, 'defref_ms_Payables', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_ShortTermBorrowings', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_ObligationToReturnSecuritiesReceivedAsCollateral', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_InterestAndDividendsPayableCurrentAndNoncurrent', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_InterestPayableCurrentAndNoncurrent', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_DividendsPayableCurrentAndNoncurrent', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_DueToAffiliateCurrentAndNoncurrent', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_AccountsPayableAndAccruedLiabilitiesCurrentAndNoncurrent', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_LiabilityForFuturePolicyBenefitsAndUnpaidClaimsAndClaimsAdjustmentExpense', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_PolicyholderContractDeposits', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_DebtCurrent', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_OtherPolicyholderFunds', window" in str(tds[0])
              ):
            if cur_liabilities == '---':
                cur_liabilities_calc = html_re(str(tds[colm]))
                if cur_liabilities_calc != '---':
                    cur_liabilities_sum += round(check_neg(str(tds[colm]), cur_liabilities_calc) * (dollar_multiplier / 1_000_000), 2)

        elif r"this, 'defref_us-gaap_Liabilities', window" in str(tds[0]):
            liabilities_calc = html_re(str(tds[colm]))
            if liabilities_calc != '---':
                liabilities = round(check_neg(str(tds[colm]), liabilities_calc) * (dollar_multiplier / 1_000_000), 2)

        elif ('this, \'defref_us-gaap_LongTermDebtNoncurrent\', window' in str(tds[0]) or
              'this, \'defref_us-gaap_LongTermDebt\', window' in str(tds[0]) or
              'this, \'defref_us-gaap_LongTermDebtAndCapitalLeaseObligations\', window' in str(tds[0]) or
              r"this, 'defref_us-gaap_LineOfCredit', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_UnsecuredLongTermDebt', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_LongTermNotesPayable', window" in str(tds[0]) or
              r"this, 'defref_stor_NonRecourseDebtNet', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_DebtInstrumentCarryingAmount', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_DebtAndCapitalLeaseObligations', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_ConvertibleDebt', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_SecuredDebt', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_SeniorNotes', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_JuniorSubordinatedDebentureOwedToUnconsolidatedSubsidiaryTrust', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_ConvertibleDebtNoncurrent', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_ConvertibleLongTermNotesPayable', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_LongTermLoansPayable', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_LongTermLineOfCredit', window" in str(tds[0]) or
              r"this, 'defref_cat_LongTermDebtDueAfterOneYearMachineryEnergyTransNoncurrent', window" in str(tds[0]) or
              r"this, 'defref_cat_LongTermDebtDueAfterOneYearFinancialProducts', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_LongTermNotesAndLoans', window" in str(tds[0]) or
              r"this, 'defref_kr_LongTermDebtAndFinanceLease', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_DebtLongtermAndShorttermCombinedAmount', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_NotesPayable', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_ConvertibleNotesPayable', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_UnsecuredDebt', window" in str(tds[0]) and 'nrz' in str(soup) or
              r"this, 'defref_nrz_ResidentialMortgageLoansRepurchaseLiability', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_LoansPayable', window" in str(tds[0]) or
              r"this, 'defref_o_NotesPayableNet', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_OtherNotesPayable', window" in str(tds[0]) or
              r"this, 'defref_o_NotesPayableNetOfUnamortizedDiscount', window" in str(tds[0]) or
              r"this, 'defref_ohi_SeniorNotesAndOtherUnsecuredBorrowingsNet', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_OtherLoansPayable', window" in str(tds[0]) or
              r"this, 'defref_ohi_TermLoan', window" in str(tds[0]) or
              r"this, 'defref_ohi_UnsecuredDebtExcludingRevolvingCreditFacilityAndTermLoan', window" in str(tds[0]) or
              r"this, 'defref_ohi_UnsecuredDebtExcludingRevolvingCreditFacility', window" in str(tds[0])
              ):
            result = html_re(str(tds[colm]))
            if result != '---':
                debt += round(result * (dollar_multiplier / 1_000_000), 2)

        elif 'this, \'defref_us-gaap_LiabilitiesAndStockholdersEquity\', window' in str(tds[0]) and tot_liabilities == '---':
            tot_liabilities_calc = html_re(str(tds[colm]))
            if tot_liabilities_calc != '---':
                tot_liabilities = round(check_neg(str(tds[colm]), tot_liabilities_calc) * (dollar_multiplier / 1_000_000), 2)

        elif ('this, \'defref_us-gaap_StockholdersEquity\', window' in str(tds[0]) and equity =='---' or
              'this, \'defref_us-gaap_StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest\', window' in str(tds[0]) and equity == '---' or
              r"this, 'defref_us-gaap_CommonStockholdersEquity', window" in str(tds[0]) and equity =='---'
              ):
            equity = html_re(str(tds[colm]))
            if equity != '---':
                equity = round(check_neg(str(tds[colm]), equity) * (dollar_multiplier / 1_000_000), 2)    

        elif ('[Member]' in str(tds[0]) or 'VIEs' in str(tds[0])) and cash != '---':
            break     

    # Use cash sum if cash total not found
    if cash == '---':
        cash = cash_sum

    # Calculate curent assets if total not found
    if cash != '---' and cur_assets == '---':
        cur_assets = round(cash + recievables, 2)

    # Calculate current liabilities if total not found
    if cur_liabilities == '---' and cur_liabilities_sum != 0:
        cur_liabilities = round(cur_liabilities_sum, 2)

    # Calculate liabilites from shareholder equity if not found
    if liabilities == '---' and tot_liabilities != '---' and equity != '---':
        liabilities = round(tot_liabilities - equity, 2)

    # Net out goodwill from assets
    if assets != '---' and goodwill != '---':
        assets = round(assets - (goodwill + sum(intangible_assets_set)), 2)
    
    return cash, cur_assets, assets, round(debt, 2), cur_liabilities, liabilities, equity

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

def cf_htm(cf_url, headers, per):
    ''' Parse .htm cash flow statement

    Args:
        URL of statement
        User agent for SEC
        Fiscal period end date
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
    cfo = capex = share_issue = buyback = divpaid = sbc = 0
    fcf = '---'
    capex_set = set()
    buyback_set = set()
    repeat_flag = False

    # Find which column has 12 month data
    colm = column_finder_annual_htm(soup, per)

    # Determine multiplier
    head = soup.find('th')
    dollar_multiplier = multiple_extractor(str(head))

    # Loop through rows, search for row of interest
    for row in soup.table.find_all('tr'):
        tds = row.find_all('td')
        if tds == []:
            continue

        # Check for companies that report subsidiares and break when that starts
        elif r"this, 'defref_us-gaap_NetCashProvidedByUsedInOperatingActivitiesAbstract', window" in str(tds[0]):
            if repeat_flag == True:
                break
            else:
                repeat_flag = True

        elif ('Net cash from operations' in str(tds[0]) or
            'this, \'defref_us-gaap_NetCashProvidedByUsedInOperatingActivities\', window' in str(tds[0]) or 
            'this, \'defref_us-gaap_NetCashProvidedByUsedInOperatingActivitiesContinuingOperations\', window' in str(tds[0])
            ): 
            cfo_calc = html_re(str(tds[colm]))
            if cfo_calc != '---':
                cfo = round(check_neg(str(tds[colm]), cfo_calc) * (dollar_multiplier / 1_000_000), 2)

        elif ('Additions to property and equipment' in str(tds[0]) or
              'this, \'defref_us-gaap_PaymentsToAcquireProductiveAssets\', window' in str(tds[0]) or
              'this, \'defref_us-gaap_PaymentsToAcquirePropertyPlantAndEquipment\', window' in str(tds[0]) or
              'this, \'defref_us-gaap_PaymentsForProceedsFromProductiveAssets\', window' in str(tds[0]) or
              r"this, 'defref_us-gaap_PaymentsToAcquireRealEstateHeldForInvestment', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_PaymentsToAcquireOtherPropertyPlantAndEquipment', window" in str(tds[0])
              ):
            capex_calc = html_re(str(tds[colm]))
            if capex_calc != '---':
                capex = round(capex_calc * (dollar_multiplier / 1_000_000), 2)
                capex_set.add(capex)

        elif ('Common stock repurchased' in str(tds[0]) or
              'this, \'defref_us-gaap_PaymentsForRepurchaseOfCommonStock\', window' in str(tds[0]) and 'Redemption of common limited partnership units' not in str(tds[0]) and 'Purchase of treasury stock' not in str(tds[0]) or
              'Shares repurchased under stock compensation plans' in str(tds[0]) or
              r"this, 'defref_us-gaap_PaymentsForRepurchaseOfEquity', window" in str(tds[0]) or
              r"this, 'defref_pld_PaymentsForRepurchaseOfPreferredStock', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_ProceedsFromIssuanceOrSaleOfEquity', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_PaymentsForRepurchaseOfPreferredStockAndPreferenceStock', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_PaymentsForRepurchaseOfCommonStock', window" in str(tds[0])
              ):
            buyback_calc = html_re(str(tds[colm]))
            if buyback_calc != '---':
                buyback = round(buyback_calc * (dollar_multiplier / 1_000_000), 2)
                buyback_set.add(buyback)

        elif ('this, \'defref_us-gaap_ProceedsFromStockOptionsExercised\', window' in str(tds[0]) or
              'this, \'defref_us-gaap_ProceedsFromIssuanceOfCommonStock\', window' in str(tds[0]) or
              'this, \'defref_us-gaap_ProceedsFromIssuanceOfSharesUnderIncentiveAndShareBasedCompensationPlansIncludingStockOptions\', window' in str(tds[0]) or
              r"this, 'defref_us-gaap_ProceedsFromIssuanceOfCommonStock', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_ProceedsFromStockPlans', window" in str(tds[0]) or
              r"this, 'defref_doc_ProceedsFromPaymentsToFromSaleOfCommonSharesNet', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_ProceedsFromIssuanceOfPreferredStockAndPreferenceStock', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_ProceedsFromSaleOfTreasuryStock', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_ProceedsFromIssuanceInitialPublicOffering', window" in str(tds[0]) or
              r"this, 'defref_ohi_NetProceedsFromIssuanceOfCommonStock', window" in str(tds[0])
              ):
            share_issue_rtn = html_re(str(tds[colm]))
            if share_issue_rtn != '---':
                share_issue += round(share_issue_rtn * (dollar_multiplier / 1_000_000), 2)

        elif ('Common stock cash dividends paid' in str(tds[0]) or
              'this, \'defref_us-gaap_PaymentsOfDividends\', window' in str(tds) and 'Dividend to former parent in connection with spin-off' not in str(tds[0]) or
              'PaymentsOfDividendsAndDividendEquivalentsOnCommonStockAndRestrictedStockUnits\', window' in str(tds[0]) or
              'this, \'defref_us-gaap_PaymentsOfDividendsCommonStock\', window' in str(tds[0]) or
              r"this, 'defref_us-gaap_Dividends', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_PaymentsOfOrdinaryDividends', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_PaymentsOfDividendsCommonStock', window" in str(tds[0]) or
              r"this, 'defref_frt_DividendsPaidToCommonAndPreferredShareholders', window" in str(tds[0])
              ):
            divpaid_calc = html_re(str(tds[colm]))
            if divpaid_calc != '---' and divpaid_calc != 0:
                divpaid = round(divpaid_calc * (dollar_multiplier / 1_000_000), 2)

        elif ('this, \'defref_us-gaap_ShareBasedCompensation\', window' in str(tds[0]) or
              'this, \'defref_us-gaap_AllocatedShareBasedCompensationExpense\', window' in str(tds[0]) or
              r"this, 'defref_epr_ShareBasedPaymentArrangementNoncashExpenseManagementAndTrustees', window" in str(tds[0]) or
              r"this, 'defref_us-gaap_EmployeeBenefitsAndShareBasedCompensation', window" in str(tds[0])
              ):
            sbc_calc = html_re(str(tds[colm]))
            if sbc_calc == '---':
                sbc_calc = html_re(str(tds[1]))
                if sbc_calc != '---':
                    colm = 1
            if sbc_calc != '---':
                sbc = round(sbc_calc * (dollar_multiplier / 1_000_000), 2)

        elif '[Member]' in str(row) and cfo != '---':
            break    

    # Calculate Free Cash Flow
    fcf = round(cfo - sum(capex_set), 2)

    # Sum buybacks, get rid of duplicates
    buyback = round(sum(buyback_set), 2)

    # Net out share issuance
    buyback = round(buyback - share_issue, 2)

    return fcf, buyback, divpaid, sbc

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

def div_htm(div_url, headers, per):
    ''' Parse various .htm docs for div data

    Args:
        URL of statement
        User agent for SEC
        Fiscal period end date
    Returns:
        Dividend (float)
    '''

    # Get data from site
    content = requests.get(div_url, headers=headers).content
    soup = BeautifulSoup(content, 'html.parser')
    colm = column_finder_annual_htm(soup, per)
    head = soup.table.find_all('tr')[0]
    
    # Initial value
    div = '---'
    
    # If company has seperate div table
    if ('EQUITY' not in str(head).upper() and
        'QUARTERLY FINANCIAL INFORMATION' not in str(head).upper() and
        'QUARTERLY RESULTS OF OPERATIONS' not in str(head).upper() and
        'UNAUDITED QUARTERLY DATA (DETAILS)' not in str(head).upper() and
        'STOCK OPTION ASSUMPTIONS' not in str(head).upper() and
        'FEDERAL INCOME TAX TREATMENT OF COMMON DIVIDENDS' not in str(head).upper() and
        'DIVIDENDS [ABSTRACT]' not in str(head).upper() and
        'SHAREHOLDERS\' EQUITY - (NARRATIVE) (DETAILS)' not in str(head).upper() and
        'SHAREHOLDERS\' EQUITY - ADDITIONAL INFORMATION (DETAIL)' not in str(head).upper() and
        'SELECTED QUARTERLY DATA (UNAUDITED) (DETAILS)' not in str(head).upper() and
        'SELECTED QUARTERLY DATA (DETAILS)' not in str(head).upper() and
        'QUARTERLY RESULTS (DETAILS)' not in str(head).upper() and
        'DIVIDENDS (Details)' not in str(head) and
        'Earnings Per Common Share Attributable to Pfizer Inc. Common Shareholders (Tables)' not in str(head) and
        'Quarterly Financial Data (Tables)' not in str(head)
        ):
        for row in soup.table.find_all('tr'):
            tds = row.find_all('td')
            if tds == []:
                continue
            # Find where quarterly data ends
            if 'onclick="top.Show.showAR( this, \'defref_us-gaap_DividendsPayableDateDeclaredDayMonthAndYear\', window )' in str(tds[0]):
                for index, row in enumerate(tds):
                    if not re.findall(r'\d', str(row)):
                        break
            if 'Distributions (Details)' not in str(head):
                if (r"this, 'defref_us-gaap_CommonStockDividendsPerShareDeclared', window" in str(tds[0]) and 'Capital Stock (Narrative) (Details) (USD $)' not in str(head) or
                    r"this, 'defref_us-gaap_CommonStockDividendsPerShareCashPaid', window" in str(tds[0])
                    ):
                    try:
                        div = html_re(str(tds[index]))
                    except:
                        div = html_re(str(tds[colm]))
                    if (div != '---' and r"this, 'defref_us-gaap_CommonStockDividendsPerShareCashPaid', window" not in str(soup) and r"this, 'defref_us-gaap_CommonStockDividendsPerShareDeclared', window" in str(tds[0]) or
                        div != '---' and div!= 0 and ('DIVIDENDS (Per Share Distributions) (Detail)' in str(head) or 'DIVIDENDS (Detail' in str(head))
                        ):
                        return div

        if 'onclick="top.Show.showAR( this, \'defref_us-gaap_CommonStockDividendsPerShareDeclared\', window )' in str(soup) and 'Distributions (Details) (USD $)' in str(soup):
            # Find index for yearly div data
            cells = soup.find_all('div')
            stock_flag = False
            div_index = 0
            for index in range(len(cells)):
                if 'Common Stock' in str(cells[index]):
                    stock_flag = True                
                else:
                    if stock_flag == True and per in str(cells[index]):
                        break
                    else:
                        stock_flag = False
                    div_index += 1

            # Find row with div data
            for row in soup.table.find_all('tr'):
                tds = row.find_all('td')
                if tds == []:
                    continue
                if r"this, 'defref_us-gaap_CommonStockDividendsPerShareCashPaid', window" in str(tds[0]):
                    div = html_re(str(tds[div_index]))
                    return div

        if div == '---':
            obj = re.findall(r'(?:\$)(\d\.\d\d)(?:</font>)', str(soup))
            if obj != []:
                div = sum(list(map(float, obj[-4:])))
                return div
        
        if div == '---' and 'Supplementary Financial Information' in str(soup):
            for row in soup.table.find_all('tr'):
                tds = row.find_all('td')
                if tds == []:
                    continue
                if r"this, 'defref_us-gaap_CommonStockDividendsPerShareCashPaid', window" in str(tds[0]):
                    div_list = re.findall(r'(?:<td class=\"nump\">\$ )(\d\d?\.\d\d?)(?:<span>)', str(tds), re.M)
                    if div_list != []:
                        div = sum(map(float, div_list[0:4]))
                        return div

        # For large dive table with preferred shares (IIPR)
        if div == '---':
            try:
                table = pd.read_html(content, match='Declaration Date')[1]
                table_filtered = table[table[2] == 'Common stock']
                div_list = list(table_filtered[5][-4:])
                div = sum(map(float, div_list))
                return div
            except:
                pass
        
        # For really weird html table
        if div =='---':
            obj = re.findall(r'(?:size=\"2\">)(\d\.\d\d\d?)(?:</font></td>)', str(soup), re.M)
            if obj != []:
                if 'For Quarter' in str(soup):
                    div = sum(map(float, obj[0:4]))
                else:
                    div = float(obj[0])

        if div != '---':
            return round(div, 3)
    
    elif ('QUARTERLY FINANCIAL INFORMATION' in str(head).upper() or
          'QUARTERLY RESULTS OF OPERATIONS' in str(head).upper() or
          'STOCK OPTION ASSUMPTIONS' in str(head).upper() or
          'UNAUDITED QUARTERLY DATA (DETAILS)' in str(head).upper() or
          'SELECTED QUARTERLY DATA (UNAUDITED) (DETAILS)' in str(head).upper() or
          'SELECTED QUARTERLY DATA (DETAILS)' in str(head).upper() or
          'QUARTERLY RESULTS (DETAILS)' in str(head).upper()
          ):
        # Find column with total data   
        tds = soup.table.find_all('tr')[5].find_all('td')
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
        
        if div != '---':
            return div

        # If data source is in one big line
        else:
            if 'UNAUDITED' not in str(head) and 'Dividends declared per common share' in str(soup):
                # Create tables with Pandas
                table = pd.read_html(content, match='Dividend')[1]               
                
                # Find row with div data and pull
                rows = table.loc[table[0] == 'Dividends declared per common share'].index
                names = list(table.iloc[0 , :])
                table.columns = names
                colm_index = table.columns.get_loc('Total Year')
                div = float(table.iloc[rows[0], colm_index])
                if div != '---':
                    return div     

        # For different table style        
        # Find div
        for row in soup.table.find_all('tr'):
            tds = row.find_all('td')
            if tds == []:
                continue
            if (r"this, 'defref_us-gaap_CommonStockDividendsPerShareCashPaid', window" in str(tds[0]) or
                r"this, 'defref_us-gaap_CommonStockDividendsPerShareDeclared', window" in str(tds[0])
                ):
                div = html_re(str(tds[colm]))
                if div is not None and div != '---':
                    return div

        # If 12 month sum not provided, but per quarter div is
        if div == '---' and '3 Months Ended' in str(head):
            for row in soup.table.find_all('tr'):
                tds = row.find_all('td')
                if tds == []:
                    continue
                if (r"this, 'defref_us-gaap_CommonStockDividendsPerShareCashPaid', window" in str(tds[0]) or
                    r"this, 'defref_us-gaap_CommonStockDividendsPerShareDeclared', window" in str(tds[0])
                    ):
                    if r'toggleNextSibling(this)' in str(tds):
                        obj = re.findall(r'(?:\">\$ )(\d?\d\.\d\d?\d?)(?:</a><span)', str(tds), re.M)
                    else:
                        obj = re.findall(r'(?:<td class=\"nump\">\$? ?)(\d?\d\.\d\d?\d?)(?:<span>)', str(tds), re.M)
                    if obj != [] and len(obj) >= 4:
                        div = sum(list(map(float, obj[:4])))
                        return round(div, 3)

    elif ("Consolidated Statements Of Changes In Stockholders' Equity (Parenthetical)" in str(head) and '12 Months Ended' in str(head) or
          "Stockholders' Equity - Dividends (Details)" in str(head) and '12 Months Ended' in str(head) or
          "Stockholders' Equity (Details 2)" in str(head) and '12 Months Ended' in str(head) or
          "Equity and Accumulated Other Comprehensive" in str(head) and '12 Months Ended' in str(head) or
          "Equity And Accumulated Other Comprehensive" in str(head) and '12 Months Ended' in str(head) or
          "Income Taxes - Federal Income Tax Treatment of Common Dividends (Details) - Common Stock" in str(head) and '12 Months Ended' in str(head) or
          "Shareholders' Equity - (Narrative) (Details)" in str(head) and '12 Months Ended' in str(head) or
          "Shareholders' Equity (Narrative) (Details)" in str(head) and '12 Months Ended' in str(head) or
          "Shareholders' Equity - Additional Information (Detail)" in str(head) and '12 Months Ended' in str(head) or
          ">Shareholders' Equity (Details 3)" in str(head) and '12 Months Ended' in str(head) or
          "Total Equity" in str(head) and '12 Months Ended' in str(head) or
          "Shareholders' Equity - Dividends Declared (Details)" in str(head) and '12 Months Ended' in str(head)
          ):      
        # Find row with div data
        for row in soup.table.find_all('tr'):
            tds = row.find_all('td')
            if tds == []:
                continue
            if (r"this, 'defref_us-gaap_CommonStockDividendsPerShareDeclared', window" in str(tds[0]) or
                r"this, 'defref_cor_DistributionsPerShare', window" in str(tds[0]) or
                r"this, 'defref_cor_DistributionsPerShare', window" in str(tds[0]) or
                r"this, 'defref_dlr_CommonStockDividendRatePerDollarAmount', window" in str(tds[0]) or
                r"this, 'defref_us-gaap_CommonStockDividendsPerShareCashPaid', window" in str(tds[0]) or
                r"this, 'defref_us-gaap_DividendsPayableAmountPerShare', window" in str(tds[0]) or
                r"this, 'defref_gs_DividendsDeclaredPerCommonShare', window" in str(tds[0])
                ):
                # Pull div
                div = html_re(str(tds[colm]))
                if div is not None and div != '---':
                    return div

        # Catch for weird format
        if 'Equity And Accumulated Other Comprehensive' in str(head) and div == '---':
            for row in soup.table.find_all('tr'):
                tds = row.find_all('td')
                if tds == []:
                    continue
                if r"this, 'defref_dlr_CommonStockDividendRatePerDollarAmount', window" in str(tds[0]):
                    # Pull div
                    result = re.search(r'(?:<td class=\"nump\">\$ )(\d?\d\.\d\d?\d?)', str(tds), re.M)
                    if result is not None:
                        div = float(result.group(1))
                        return div

    elif 'Stockholders\' Equity (Dividends) (Details)' in str(head) and '0 Months Ended' in str(head):
        spec_div = '---'
        # Find correct cell index for div and special div
        index = 4
        spec_index = 0
        if '<div>Special Cash Dividend</div>' in str(soup):
            for cell in soup.table.find_all('div'):
                    if 'Special Cash Dividend' in str(cell):
                        index += 1
                    elif per in str(cell):
                        spec_index = index
                    elif 'Ordinary Dividend' in str(cell):
                        break
        else:
            index_list = []
            head = soup.table.find_all('tr')[1]
            for index, cell in enumerate(head.find_all('div')):
                if per in str(cell):
                    index_list.append(1 + index)

        # Find div data
        if r'onclick="toggleNextSibling(this);' in str(soup):
            for row in soup.table.find_all('tr'):
                tds = row.find_all('td')
                if tds == []:
                    continue
                if (r"this, 'defref_us-gaap_CommonStockDividendsPerShareDeclared', window" in str(tds[0]) or
                    r"this, 'defref_us-gaap_CommonStockDividendsPerShareCashPaid', window" in str(tds[0])
                    ):
                    # Find special div if present
                    if spec_div == '---':
                        spec_result = re.search(r'(?:\">\$ )(\d?\d\.\d\d?\d?)', str(tds[index_list[0]]))
                        if spec_result != None:
                            spec_div = float(spec_result.group(1))

                    # Find div if present
                    if div == '---':
                        result = re.search(r'(?:\">\$ )(\d?\d\.\d\d?\d?)', str(tds[index_list[1]]))
                        if result != None:
                            result = re.findall(r'(?:\">\$ )(\d?\d\.\d\d?\d?)', str(tds))
                            div = round(sum(list(map(float, result[:4]))), 3)
                            if spec_div != '---':
                                div += spec_div
                                return div

        else:
            for row in soup.table.find_all('tr'):
                tds = row.find_all('td')
                if tds == []:
                    continue
                if (r"this, 'defref_us-gaap_CommonStockDividendsPerShareDeclared', window" in str(tds[0]) or
                    r"this, 'defref_us-gaap_CommonStockDividendsPerShareCashPaid', window" in str(tds[0])
                    ):
                    # Find special div if present
                    spec_result = re.search(r'(?:<td class=\"nump\">\$ )(\d?\d\.\d\d?\d?)(?:<span>)', str(tds[spec_index]))
                    if spec_result != None:
                        spec_div = float(spec_result.group(1))

                    # Find div
                    result = re.findall(r'(?:<td class=\"nump\">\$ )(\d?\d\.\d\d?\d?)(?:<span>)', str(tds[index:]), re.M)
                    if result is not []:
                        div = sum(list(map(float, result[:4])))
                        if spec_result != None:
                            div += spec_div
                        return div

    # If company is a jerk and burries it (like apple)
    else:
        # If contained in equity increase/decrease statement
        if ('defref_us-gaap_IncreaseDecreaseInStockholdersEquityRollForward' in str(soup) or
            'Balance at, beginning of period at' in str(soup)
            ):
            for row in soup.table.find_all('tr'):
                tds = row.find_all('td')
                if tds == []:
                    continue
                if ('this, \'defref_us-gaap_CommonStockDividendsPerShareDeclared\', window' in str(tds[0]) or
                    r"this, 'defref_us-gaap_CommonStockDividendsPerShareCashPaid', window" in str(tds[0])
                    ):
                    div = float(re.findall(r'\d+\.\d+', str(tds))[0])
            return div
        
        # If no 12 month data, for divs broken out by quarter
        elif ('12 Months Ended' not in str(head) and 'ABSTRACT' not in str(head).upper() or
              'Stockholders\' equity (Details Textual)' in str(head) and '3 Months Ended' in str(head) or
              'Stockholders\' equity (Details Textual)' in str(head) and '0 Months Ended' in str(head) or
              'Shareholders\' Equity - Dividends Declared (Details' in str(head) and '3 Months Ended' in str(head)
              ):
            if ('this, \'defref_us-gaap_CommonStockDividendsPerShareDeclared\', window' in str(soup) or
                r"this, 'defref_us-gaap_CommonStockDividendsPerShareCashPaid', window" in str(soup)
                ):
                period_found = False
                new_year = int(per[-4:]) - 1
                new_per = per[:-4] + str(new_year)
                for row in soup.table.find_all('tr'):
                    tds = row.find_all('td')
                    if new_per in str(row):
                        period_found = True
                    if tds == []:
                        continue
                    elif ('this, \'defref_us-gaap_CommonStockDividendsPerShareDeclared\', window' in str(tds[0]) and period_found == True or
                          r"this, 'defref_us-gaap_CommonStockDividendsPerShareCashPaid', window" in str(tds[0]) and period_found == True
                          ):
                        if html_re(str(tds[colm])) == '---':
                            div = re.findall(r'\d+\.\d+', str(tds))[0:4]
                        else:
                            div = re.findall(r'\d+\.\d+', str(tds[colm:colm + 4]))[0:4]
                        div = sum(list(map(float, div)))
                        return round(div, 3)
            elif '>CONSOLIDATED STATEMENT OF SHAREOWNERS EQUITY - USD ($)' in str(head):
                period_found = False
                for row in soup.table.find_all('tr'):
                    if per in str(row):
                        period_found = True
                    elif r"this, 'defref_us-gaap_CommonStockDividendsPerShareCashPaid', window" in str(row) and period_found == True:
                        tds = row.find_all('td')
                        result = re.search(r'\d+\.\d+', str(tds[1]))
                        div = float(result.group(0))
                        return div

        # If data is not quarterly
        elif 'QUARTERLY' not in soup.text.upper() and div == '---' and '>Record Date<' not in str(soup):
            # Check for three month data prior to 12 month data
            if '4 Months Ended' in str(soup):

                # Check if 3 month data appears earlier
                four_mon = re.search(r'4 Months Ended', str(head), re.M)
                twelve_mon = re.search(r'12 Months Ended', str(head), re.M)
                if four_mon is not None and twelve_mon is not None:
                    if four_mon.span()[0] < twelve_mon.span()[0]:
                        # Find column where 12 month data starts
                        colm = re.findall(r'(?:colspan=\"(\d)\")(?!>12 Months Ended)', str(head), re.M)
                        colm = sum(map(int, colm))
            
            spec_div = False
            if ('Special cash dividend' in str(soup) or
                'Special dividend declared' in str(soup)
                ):
                spec_div = True
            for row in soup.table.find_all('tr'):
                tds = row.find_all('td')
                if tds == []:
                    continue
                if ('this, \'defref_us-gaap_CommonStockDividendsPerShareDeclared\', window' in str(tds[0]) or
                    'this, \'defref_us-gaap_CommonStockDividendsPerShareCashPaid\', window' in str(tds[0]) or
                    r"this, 'defref_cor_DistributionsPerShare', window" in str(tds[0]) or
                    r"this, 'defref_nem_DividendsPayableCashAmountPerShare', window" in str(tds[0])
                    ):
                    div_result = html_re(str(tds[colm]))
                    if div_result != '---':
                        div = div_result
                    if spec_div == False and div != '---':
                        return div
                    else:
                        continue
                if spec_div == True:        
                    if (r"this, 'defref_ldos_SpecialDividendPerShareDeclared', window" in str(tds[0]) or
                        r"this, 'defref_nem_DividendsPayableSpecialAmountPerShare', window" in str(tds[0])
                        ):
                        spec_div_float = html_re(str(tds[colm]))
                        if div != '---' and spec_div_float != '---':
                            return round(div + spec_div_float, 3)
        
        else:
            # Find multiplier
            multiplier = 1
            for header in soup.table.find_all('th'):
                if '9 Months Ended' in str(header):
                    multiplier = 3
                    break

                elif '12 Months Ended' in str(header):
                    multiplier = 4
                    break

            # Find div data
            for row in soup.table.find_all('tr'):
                tds = row.find_all('td')
                if tds == []:
                    continue
                if ('this, \'defref_v_Cashdividendsdeclaredandpaidquarterlyperasconvertedshare\', window' in str(tds[0]) or
                    'this, \'defref_us-gaap_CommonStockDividendsPerShareDeclared\', window' in str(tds[0]) and 'quarterly' in str(tds[0])):
                    div = multiplier * html_re(str(tds[colm]))
                    if multiplier == 3:
                        div += float(''.join(re.findall(r'\d+\.\d+', tds[1].text.strip())))

    # If data source is in one big line
    if div == '---':

        # Check if listed in big line on equity report
        if '>Cash dividends declared per share: $' in str(soup) and 'Shareholders\' Equity (Tables)' in str(head):
            try:
                obj = float(re.findall(r'(?:Cash dividends declared per share: .)(\d?\d\.\d\d?)', str(soup))[-1])
                if obj is not None:
                    div = obj
                    return div
            except:
                pass

        # Create tables with Pandas
        tables = soup.find_all('table')[0] 
        panda = pd.read_html(str(tables))

        # Look for table with div data
        for table in panda:
            if ('Dividends Per Share' in table.values or
                'DividendsPer Share' in table.values or
                'Cash dividends per share' in table.values or
                'Cash dividends declared per share' in table.values or
                'Dividends declared' in table.values
                ):                    
                # Find row with div data and pull
                row_list = ['Cash dividends per share', 'Cash dividends declared per share', 'Dividends declared']
                for name in row_list:
                    row = table.loc[table[0] == name]
                    if not row.empty:
                        break
                if 'Quarterly Financial Data (Tables)' in str(head):
                    div_sum = re.findall(r'\d+\.\d\d', str(row))[0:4]
                    div = round(sum(map(float, div_sum)), 3)
                else:
                    div = float(re.findall(r'\d+\.\d\d', str(row))[0])
                if div != '---':
                    return div
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
                    try:
                        answer = table.at[row[0] + 1, col + 2]
                        div = float(answer)
                        break
                    except:
                        continue
                break

        if div == '---' and 'Share Repurchase Program' not in str(head) and 'Shareholders\' Equity' not in str(head):
            try:
                table = pd.read_html(content, match='Dividend')[1]
                div_list = list(table[3][-4:])
                div = round(sum(map(float, div_list)), 3)
                if math.isnan(div):
                    div = '---'
            except:
                pass
    
    return div


def eps_catch_htm(catch_url, headers, per):
    ''' Parse EPS and div info if not listed elsewhere

    Args:
        URL of statement
        User agent for SEC
        Fiscal period end date
    Returns:
        EPS (float)
        Dividend (float)
    '''
    
    # Initial values
    div = eps = shares = '---'

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

    # Final catch for EPS data
    if eps == '---':
        colm = column_finder_annual_htm(soup, per)
        head = soup.find('th')
        dollar_multiplier, share_multiplier = multiple_extractor(str(head), shares=True)
        for row in soup.table.find_all('tr'):
            tds = row.find_all('td')
            if tds == []:
                continue

            if ('EarningsPerShareDiluted' in str(tds[0]) and eps == '---' or
                'this, \'defref_us-gaap_EarningsPerShareBasicAndDiluted\', window' in str(tds[0]) and eps == '---' or
                r"this, 'defref_us-gaap_IncomeLossFromContinuingOperationsPerBasicAndDilutedShare', window" in str(tds[0]) and eps == '---' or
                r"this, 'defref_us-gaap_IncomeLossFromContinuingOperationsPerDilutedShare', window" in str(tds[0]) and eps == '---'
                ):
                result = html_re(str(tds[colm]))
                if result != '---':
                    eps = check_neg(str(tds[colm]), result)

            elif (r"this, 'defref_us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding', window" in str(tds[0]) or
                  r"this, 'defref_us-gaap_WeightedAverageNumberOfShareOutstandingBasicAndDiluted', window" in str(tds[0]) or
                  r"this, 'defref_tsla_WeightedAverageNumberOfSharesOutstandingBasicAndDilutedOne', window" in str(tds[0])
                  ):
                shares_calc = html_re(str(tds[colm]))
                if shares_calc != '---':
                    shares = round(shares_calc * (share_multiplier / 1_000), 2)


    return eps, div, shares


'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

def share_catch_htm(catch_url, headers, per):
    ''' Parse share data if not found in income statement

    Args:
        URL of statement
        User agent for SEC
        Fiscal period end date
    Returns:
        Shares (int)     
    '''

    # Get data from site
    content = requests.get(catch_url, headers=headers).content
    soup = BeautifulSoup(content, 'html.parser')
    header = str(soup.table.find_all('tr')[0])

    # Initial values
    shares = '---'
    shares_issued = shares_repurchased = 0
    share_set = set()
    shares_issued_set = set()
    shares_repurchased_set = set()

    # Find which column has 12 month data
    colm = column_finder_annual_htm(soup, per)

    # Determine multiplier
    head = soup.find('th')
    dollar_multiplier, share_multiplier = multiple_extractor(str(head), shares=True)

    # Loop through rows, search for row of interest
    if ('Equity (Common Stock and Class B Stock Changes in Number of Shares Issued, Held in Treasury and Outstanding) (Details)' in header or
        'Equity (Common Stock Changes in Number of Shares Issued, Held in Treasury and Outstanding) (Details)' in header
        ):
        if '1 Months Ended' in header and "Increase (Decrease) in Stockholders' Equity [Roll Forward]" not in str(soup):
            for row in soup.table.find_all('tr'):
                tds = row.find_all('td')
                if tds == []:
                    continue

                elif r"this, 'defref_us-gaap_CommonStockSharesIssued', window" in str(tds[0]) and 'Ending Balance' in str(tds[0]):
                    col_list = [12, 15]
                    for colm_guess in col_list:
                        shares_calc = html_re(str(tds[colm_guess]))
                        if shares_calc != '---':
                            shares = round(shares_calc * (share_multiplier / 1_000), 2)
                            return shares
            
        else:
            common_section = False
            for row in soup.table.find_all('tr'):
                tds = row.find_all('td')
                if tds == []:
                    continue

                elif common_section == False and 'Outstanding' in str(tds[0]):
                    common_section = True

                elif common_section == True and r"this, 'defref_us-gaap_CommonStockSharesIssued', window" in str(tds[0]) and 'Ending Balance' in str(tds[0]):
                    shares_calc = html_re(str(tds[colm]))
                    if shares_calc != '---':
                        shares = round(shares_calc * (share_multiplier / 1_000), 2)
                        return shares

    else:
        for row in soup.table.find_all('tr'):
            tds = row.find_all('td')
            if tds == []:
                continue

            if (r"this, 'defref_us-gaap_CommonStockSharesOutstanding', window" in str(tds[0]) or
                r"this, 'defref_us-gaap_CommonStockValueOutstanding', window" in str(tds[0]) or
                r"this, 'defref_us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding', window" in str(tds[0])
                ):
                shares_calc = html_re(str(tds[colm]))
                if shares_calc != '---':
                    shares = round(shares_calc * (share_multiplier / 1_000), 2)
                    share_set.add(shares)

            elif (r"this, 'defref_us-gaap_CommonStockSharesIssued', window" in str(tds[0])
                ):
                shares_issued_calc = html_re(str(tds[colm]))
                if shares_issued_calc != '---':
                    shares_issued = round(shares_issued_calc * (share_multiplier / 1_000), 2)
                    shares_issued_set.add(shares_issued)

            elif (r"this, 'defref_us-gaap_TreasuryStockShares', window" in str(tds[0])
                ):
                shares_repurchased_calc = html_re(str(tds[colm]))
                if shares_repurchased_calc != '---':
                    shares_repurchased = round(shares_repurchased_calc * (share_multiplier / 1_000), 2)
                    shares_repurchased_set.add(shares_repurchased)

        shares = round(sum(share_set), 2)
        if shares == 0:
            shares = '---'
        shares_issued = sum(shares_issued_set)
        shares_repurchased = sum(shares_repurchased_set)

        # Calculate share if outstanding not given
        if shares == '---' and shares_issued + shares_repurchased > 0:
            shares = round(shares_issued - shares_repurchased, 2)
            if shares < 0:
                shares = shares_issued

    return shares

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
        if '<Id>1</Id><IsAbstractGroupTitle>false</IsAbstractGroupTitle>' not in str(soup):
            colm = re.search(r'(?:<Id>(\d\d?)</Id>\n)(?:<Labels>\n<Label Id=\"1\" Label=\"12 Months Ended\"/>)', str(soup), re.M)
            try:
                return int(colm.group(1)) - 1
            except:
                return 0
        else:
            colm = re.search(r'(?:Label=\"12 Months Ended\"/>)(?:.|\n)*?(?:<Column><Id>(\d\d?)</Id>)', str(soup), re.M)
            try:
                return int(colm.group(1)) - 2
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
        Fiscal period end date
        Name of company       
    '''

    # Get data from site
    content = requests.get(sum_url, headers=headers).content
    soup = BeautifulSoup(content, 'xml')
    
    # Initial values
    fy = period_end = name = '---'

    # Loop through rows, search for FY and period end date
    rows = soup.find_all('Row')
    for row in rows:
        if r'DocumentFiscalYearFocus' in str(row):
            obj = re.search(r'(?:<NonNumbericText>)(\d+)(?:</NonNumbericText>)', str(row), re.M)
            fy = int(obj.group(1))
        elif r'dei_DocumentPeriodEndDate' in str(row):
            obj = re.search(r'(?:<NonNumbericText>)(.*?)(?:</NonNumbericText>)', str(row), re.M)
            period_end = datetime.strptime(obj.group(1), '%Y-%m-%d')
            period_end = period_end.strftime('%b. %d, %Y')
            if 'May' in period_end:
                period_end = period_end.replace('May.', 'May')
            if ', 0' in period_end:
                period_end = period_end.replace(', 0.', ', ')
        if r'dei_EntityRegistrantName' in str(row):
            obj = re.search(r'(?:<NonNumbericText>)(.*)(?:</NonNumbericText>)', str(row), re.M)
            name = str(obj.group(1).strip())

    if fy != '---' and period_end == '---':
        colms = soup.find_all('Column')
        re_string = '\d?\d/\d?\d/' + str(fy)
        obj = re.findall(re_string, str(colms[0].MCU.KeyName), re.M)
        period_end = datetime.strptime(obj[1], '%m/%d/%Y')
        period_end = period_end.strftime('%b. %d, %Y')
        if 'May' in period_end:
            period_end = period_end.replace('May.', 'May')
        if ', 0' in period_end:
            period_end = period_end.replace(', 0.', ', ')    

    return fy, period_end, name

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
    gross = oi = net = eps = shares = div = '---'
    rev = cost = cost_sum = share_sum = research = op_exp = dep_am = impairment = disposition = ffo = 0
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
        
        if (r'us-gaap_Revenues<' in str(row.ElementName) or
            r'us-gaap_SalesRevenueNet<' in str(row.ElementName) or
            r'us-gaap_SalesRevenueGoodsNet<' in str(row.ElementName) or
            r'us-gaap_ElectricUtilityRevenue<' in str(row.ElementName) or
            r'us-gaap_UtilityRevenue<' in str(row.ElementName) or
            r'us-gaap_RealEstateRevenueNet<' in str(row.ElementName) or
            r'gs_NetRevenuesIncludingNetInterestIncome<' in str(row.ElementName) or
            r'us-gaap_SalesRevenueServicesNet<' in str(row.ElementName) or
            r'us-gaap_RevenueMineralSales<' in str(row.ElementName)
            ):
            rev = round(xml_re(str(cells[colm].RoundedNumericAmount)) * (dollar_multiplier / 1_000_000), 2)

        elif r'us-gaap_GrossProfit<' in str(row.ElementName):
            result = xml_re(str(cells[colm].RoundedNumericAmount))
            if result != '---':
                gross = round(check_neg(str(cells[colm].RoundedNumericAmount), result, 'xml')    * (dollar_multiplier / 1_000_000), 2)  

        elif (r'us-gaap_CostOfRevenue<' in str(row.ElementName) and cost == 0 or
              r'us-gaap_CostOfGoodsAndServicesSold<' in str(row.ElementName) and cost == 0 or
              r'us-gaap_CostOfGoodsSold<' in str(row.ElementName) and cost == 0 or
              r'CostOfGoodsSoldExcludingAmortizationOfAcquiredIntangibleAssets<' in str(row.ElementName) and cost == 0 or
              r'us-gaap_CostOfGoodsAndServicesEnergyCommoditiesAndServices<' in str(row.ElementName) and cost == 0 or
              r'gs_BrokerageClearingExchangeAndDistributionFees<' in str(row.ElementName) and cost == 0 or
              r'irm_CostOfSalesExcludingDepreciationAndAmortization<' in str(row.ElementName) and cost == 0 or
              r'us-gaap_DirectOperatingCosts<' in str(row.ElementName) and cost == 0 or
              r'us-gaap_MineralExtractionProcessingAndMarketingCosts<' in str(row.ElementName) and cost == 0 or
              r'us-gaap_InterestExpense<' in str(row.ElementName) and cost == 0
              ):
            cost = round(xml_re(str(cells[colm].RoundedNumericAmount)) * (dollar_multiplier / 1_000_000), 2) 

        elif (r'us-gaap_PolicyholderBenefitsAndClaimsIncurredNet<' in str(row.ElementName) and cost == 0 or
              r'us-gaap_InterestCreditedToPolicyholdersAccountBalances<' in str(row.ElementName) and cost == 0 or
              r'us-gaap_PolicyholderDividends<' in str(row.ElementName) and cost == 0
              ):
            result = xml_re(str(cells[colm].RoundedNumericAmount))
            if result != '---':
                cost_sum += round(result * (dollar_multiplier / 1_000_000), 2)

        elif (r'us-gaap_ResearchAndDevelopmentExpense<' in str(row.ElementName) or
              r'us-gaap_ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost' in str(row.ElementName)
              ):
            research = round(xml_re(str(cells[colm].RoundedNumericAmount)) * (dollar_multiplier / 1_000_000), 2)  

        elif (r'de_CostsAndExpensesIncludingInterest<' in str(row.ElementName) or
              r'us-gaap_NoninterestExpense<' in str(row.ElementName) or
              r'o_TotalExpenses<' in str(row.ElementName) or
              r'us-gaap_BenefitsLossesAndExpenses<' in str(row.ElementName)
              ):
            op_exp = round(xml_re(str(cells[colm].RoundedNumericAmount)) * (dollar_multiplier / 1_000_000), 2) 

        elif (r'us-gaap_OperatingIncomeLoss<' in str(row.ElementName) or
              r'us-gaap_IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments<' in str(row.ElementName) and oi == '---' or
              r'us-gaap_OperatingIncomeLoss<' in str(row.ElementName) or
              r'hrl_OperatingIncomeLossIncludingEquityMethodInvestments<' in str(row.ElementName)
              ):
            result = xml_re(str(cells[colm].RoundedNumericAmount))
            if result != '---':
                oi = round(check_neg(str(cells[colm].RoundedNumericAmount), result, 'xml') * (dollar_multiplier / 1_000_000), 2)  

        elif (r'us-gaap_NetIncomeLoss<' in str(row.ElementName) and net_check == False or
              r'us-gaap_ProfitLoss<' in str(row.ElementName) and net == '---' and 'including non-controlling interest' not in str(row) and 'including noncontrolling interests' not in str(row) or
              r'us-gaap_NetIncomeLossAvailableToCommonStockholdersBasic<' in str(row.ElementName) and ('Net earnings attributable to common stockholders' in str(row) or 'Net income available to common stockholders' in str(row))
              ):
            result = xml_re(str(cells[colm].RoundedNumericAmount))
            if result != '---':
                net = round(check_neg(str(cells[colm].RoundedNumericAmount), result, 'xml') * (dollar_multiplier / 1_000_000), 2)    
            if r'us-gaap_NetIncomeLoss<' in str(row.ElementName):
                net_check = True

        elif (r'us-gaap_EarningsPerShareDiluted<' in str(row.ElementName) and eps == '---' or
              r'o_Diluted<' in str(row.ElementName) and eps == '---'
              ):
            result = xml_re(str(cells[colm].RoundedNumericAmount))
            if result != '---':
                eps = check_neg(str(cells[colm].RoundedNumericAmount), result, 'xml')

        elif r'us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding<' in str(row.ElementName):
            shares = round(xml_re(str(cells[colm].RoundedNumericAmount)) * (share_multiplier / 1_000))
            share_sum = float(round(share_sum + shares, 2))

        elif (r'us-gaap_CommonStockDividendsPerShareCashPaid<' in str(row.ElementName) or
              r'us-gaap_CommonStockDividendsPerShareDeclared<' in str(row.ElementName)
              ):
            div = xml_re(str(cells[colm].RoundedNumericAmount)) 

        elif (r'us-gaap_DepreciationAmortizationAndAccretionNet<' in str(row.ElementName) or
              r'us-gaap_DepreciationDepletionAndAmortization<' in str(row.ElementName) or
              r'us-gaap_DepreciationAndAmortization<' in str(row.ElementName)
              ):
            dep_am = round(xml_re(str(cells[colm].RoundedNumericAmount)) * (dollar_multiplier / 1_000_000), 2)  

        elif (r'us-gaap_GoodwillImpairmentLoss<' in str(row.ElementName) or
              r'us-gaap_AssetImpairmentCharges<' in str(row.ElementName) or
              r'us-gaap_ImpairmentOfRetainedInterest<' in str(row.ElementName) or
              r'us-gaap_ImpairmentOfRealEstate<' in str(row.ElementName)
              ):  
            impairment += round(xml_re(str(cells[colm].RoundedNumericAmount)) * (dollar_multiplier / 1_000_000), 2)   

        elif (r'irm_GainLossOnDispositionAndImpairmentOfAssets<' in str(row.ElementName) or
              r'us-gaap_GainLossOnDispositionOfOtherAssets<' in str(row.ElementName)
              ): 
            result = xml_re(str(cells[colm].RoundedNumericAmount))
            if result != '---':
                disposition = round(check_neg(str(row), result, 'xml') * (dollar_multiplier / 1_000_000), 2)  

    # Calculate gross if not listed
    if gross == '---':
        gross = round(rev - max(cost, cost_sum), 2)

    # Calculate operating income if not found
    if oi == '---':
        oi = round(rev - op_exp, 2)

    # Calculate EPS if not listed
    if eps == '---' and net != '---' and share_sum != 0:
        eps = round(net / share_sum, 2)

    if share_sum == 0:
        share_sum = '---'

    # Calculate FFO for REITS
    if net != '---':
        ffo = round(net + dep_am + impairment - disposition, 2)

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
    equity = cash = cur_assets = assets = cur_liabilities = liabilities = '---'
    cash_sum = recievables = intangible_assets = goodwill = debt = cur_liabilities_sum = 0
    debt_set = set()

    # Find which column has 12 month data
    colm = column_finder_annual_xml(soup)
    
    # Determine multiplier
    head = soup.RoundingOption
    dollar_multiplier = multiple_extractor(str(head), xml=True)

    # Loop through rows, search for row of interest
    rows = soup.find_all('Row')
    for row in rows:
        cells = row.find_all('Cell')
        
        if ('us-gaap_CashCashEquivalentsAndShortTermInvestments<' in str(row.ElementName) or
            'us-gaap_CashAndCashEquivalentsAtCarryingValue<' in str(row.ElementName) or
            r'us-gaap_Cash<' in str(row.ElementName) or
            r'us-gaap_CashEquivalentsAtCarryingValue<' in str(row.ElementName) or
            r'gs_CashAndCashEquivalents<' in str(row.ElementName)
            ):
            cash = round(xml_re(str(cells[colm].RoundedNumericAmount)) * (dollar_multiplier / 1_000_000), 2) 

        elif (r'us-gaap_CashAndDueFromBanks<' in str(row.ElementName) or
              r'us-gaap_InterestBearingDepositsInBanks<' in str(row.ElementName) or
              r'us-gaap_CashReserveDepositRequiredAndMade<' in str(row.ElementName)
              ):
            cash_calc = xml_re(str(cells[colm].RoundedNumericAmount))
            if cash_calc != '---':
                cash_sum += round(cash_calc * (dollar_multiplier / 1_000_000), 2)

        elif 'us-gaap_Goodwill' in str(row.ElementName):
            goodwill = round(xml_re(str(cells[colm].RoundedNumericAmount)) * (dollar_multiplier / 1_000_000), 2)  

        elif r'us-gaap_IntangibleAssetsNetExcludingGoodwill<' in str(row.ElementName):
            intangible_assets = round(xml_re(str(cells[colm].RoundedNumericAmount)) * (dollar_multiplier / 1_000_000), 2)

        elif r'us-gaap_Assets<' in str(row.ElementName):
            assets = round(xml_re(str(cells[colm].RoundedNumericAmount)) * (dollar_multiplier / 1_000_000), 2)   

        elif r'us-gaap_AssetsCurrent<' in str(row.ElementName):
            cur_assets = round(xml_re(str(cells[colm].RoundedNumericAmount)) * (dollar_multiplier / 1_000_000), 2) 
        
        elif (r'us-gaap_AccountsReceivableNet<' in str(row.ElementName) or
              r'gs_CashAndSecuritiesSegregatedForRegulatoryAndOtherPurposes<' in str(row.ElementName) or
              r'gs_SecuritiesPurchasedUnderAgreementsToResellAndFederalFundsSold<' in str(row.ElementName) or
              r'us-gaap_SecuritiesBorrowed<' in str(row.ElementName) or
              r'gs_ReceivablesFromBrokersDealersAndClearingOrganizationsBS<' in str(row.ElementName) or
              r'gs_ReceivablesFromCustomersAndCounterparties<' in str(row.ElementName) or
              r'us-gaap_ReceivablesFromBrokersDealersAndClearingOrganizations<' in str(row.ElementName) or
              r'us-gaap_AccruedFeesAndOtherRevenueReceivable<' in str(row.ElementName) or
              r'us-gaap_NotesAndLoansReceivableNetNoncurrent<' in str(row.ElementName) or
              r'us-gaap_FederalFundsSoldAndSecuritiesPurchasedUnderAgreementsToResell<' in str(row.ElementName) or
              r'us-gaap_SecuritiesReceivedAsCollateral<' in str(row.ElementName) or
              r'us-gaap_LoansAndLeasesReceivableNetReportedAmount<' in str(row.ElementName) or
              r'us-gaap_InterestsContinuedToBeHeldByTransferorFairValue<' in str(row.ElementName) or
              r'us-gaap_AccountsAndNotesReceivableNet<' in str(row.ElementName) or
              r'us-gaap_DeferredRentReceivablesNet<' in str(row.ElementName)
              ):
            recievables_calc = xml_re(str(cells[colm].RoundedNumericAmount))
            if recievables_calc != '---':
                recievables += round(recievables_calc * (dollar_multiplier / 1_000_000), 2) 

        elif (r'us-gaap_LongTermDebtNoncurrent<' in str(row.ElementName) or
              r'us-gaap_LongTermDebtAndCapitalLeaseObligations<' in str(row.ElementName) or
              r'us-gaap_LongTermDebt<' in str(row.ElementName) or
              r'us-gaap_LongTermLoansPayable<' in str(row.ElementName) or
              r'cat_LongTermDebtDueAfterOneYearMachineryAndEnginesNoncurrent<' in str(row.ElementName) or
              r'cat_LongTermDebtDueAfterOneYearFinancialProducts<' in str(row.ElementName) or
              r'us-gaap_UnsecuredLongTermDebt<' in str(row.ElementName) or
              r'us-gaap_LongTermNotesAndLoans<' in str(row.ElementName) or
              r'us-gaap_LineOfCredit<' in str(row.ElementName) or
              r'us-gaap_SecuredDebt<' in str(row.ElementName) or
              r'us-gaap_ConvertibleNotesPayable<' in str(row.ElementName) or
              r'us-gaap_NotesPayable<' in str(row.ElementName) or
              r'us-gaap_LongTermDebtComponentsNotesPayable<' in str(row.ElementName)
              ):
            debt_calc = xml_re(str(cells[colm].RoundedNumericAmount))
            if debt_calc != '---':
                debt_set.add(round(debt_calc * (dollar_multiplier / 1_000_000), 2))

        elif r'us-gaap_LiabilitiesCurrent<' in str(row.ElementName):
            cur_liabilities = round(xml_re(str(cells[colm].RoundedNumericAmount)) * (dollar_multiplier / 1_000_000), 2)

        elif (r'us-gaap_AccountsPayableAndAccruedLiabilitiesCurrentAndNoncurrent<' in str(row.ElementName) and cur_liabilities == '---' or
              r'us-gaap_UnsecuredDebt<' in str(row.ElementName) and cur_liabilities == '---' or
              r'us-gaap_Deposits<' in str(row.ElementName) and cur_liabilities == '---' or
              r'us-gaap_SecuritiesSoldUnderAgreementsToRepurchase<' in str(row.ElementName) and cur_liabilities == '---' or
              r'gs_SecuritiesLoanedBS<' in str(row.ElementName) and cur_liabilities == '---' or
              r'us-gaap_OtherSecuredFinancings<' in str(row.ElementName) and cur_liabilities == '---' or
              r'gs_PayablesToBrokerDealersAndClearingOrganizationsBS<' in str(row.ElementName) and cur_liabilities == '---' or
              r'gs_PayablesToCustomersAndCounterparties<' in str(row.ElementName) and cur_liabilities == '---' or
              r'us-gaap_FinancialInstrumentsSoldNotYetPurchasedAtFairValue<' in str(row.ElementName) and cur_liabilities == '---' or
              r'gs_UnsecuredShortTermBorrowingsIncludingCurrentPortionOfUnsecuredLongTermBorrowings<' in str(row.ElementName) and cur_liabilities == '---' or
              r'gs_OtherLiabilitiesAndAccruedExpenses<' in str(row.ElementName) and cur_liabilities == '---' or
              r'us-gaap_ShortTermBorrowings<' in str(row.ElementName) and cur_liabilities == '---' or
              r'us-gaap_ObligationToReturnSecuritiesReceivedAsCollateral<' in str(row.ElementName) and cur_liabilities == '---' or
              r'us-gaap_SecuritiesLoaned<' in str(row.ElementName) and cur_liabilities == '---' or
              r'us-gaap_AccountsPayableCurrentAndNoncurrent<' in str(row.ElementName) and cur_liabilities == '---' or
              r'us-gaap_PayablesToBrokerDealersAndClearingOrganizations<' in str(row.ElementName) and cur_liabilities == '---' or
              r'us-gaap_InterestAndDividendsPayableCurrentAndNoncurrent<' in str(row.ElementName) and cur_liabilities == '---' or
              r'us-gaap_InterestPayableCurrentAndNoncurrent<' in str(row.ElementName) and cur_liabilities == '---' or
              r'us-gaap_LiabilityForFuturePolicyBenefitsAndUnpaidClaimsAndClaimsAdjustmentExpense<' in str(row.ElementName) and cur_liabilities == '---' or
              r'us-gaap_OtherPolicyholderFunds' in str(row.ElementName) and cur_liabilities == '---'
              ):
            liabilities_calc = xml_re(str(cells[colm].RoundedNumericAmount))
            if liabilities_calc != '---':              
                cur_liabilities_sum += round(liabilities_calc * (dollar_multiplier / 1_000_000), 2)

        elif r'us-gaap_Liabilities<' in str(row.ElementName):
            liabilities = round(xml_re(str(cells[colm].RoundedNumericAmount)) * (dollar_multiplier / 1_000_000), 2)

        elif r'us-gaap_LiabilitiesAndStockholdersEquity<' in str(row.ElementName) and liabilities == '---':
            tot_liabilities = round(xml_re(str(cells[colm].RoundedNumericAmount)) * (dollar_multiplier / 1_000_000), 2)

        elif (r'us-gaap_StockholdersEquity<' in str(row.ElementName) or
              r'us-gaap_StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest<' in str(row.ElementName) and equity == '---'
              ):
            equity = xml_re(str(cells[colm].RoundedNumericAmount)) 
            if equity != '---':
                equity = round(check_neg(str(row), equity, 'xml') * (dollar_multiplier / 1_000_000), 2)                                                 

    # Use cash sum if cash total not found
    if cash == '---':
        cash = cash_sum

    # Calculate curent assets if total not found
    if cash != '---' and cur_assets == '---':
        cur_assets = round(cash + recievables, 2)

    # Calculate debt total
    debt = round(sum(debt_set), 2)

    # Calculate current liabilities if total not found
    if cur_liabilities == '---' and cur_liabilities_sum != 0:
        cur_liabilities = round(cur_liabilities_sum, 2)

    # Calculate liabilites from shareholder equity if not found
    if liabilities == '---' and tot_liabilities != '---' and equity != '---':
        liabilities = round(tot_liabilities - equity, 2)

    # Net out goodwill from assets
    if assets != '---' and goodwill != '---':
        assets = round(assets - (goodwill + intangible_assets), 2)
    
    return cash, cur_assets, assets, debt, cur_liabilities, liabilities, equity

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
    capex = share_issue = buyback = divpaid = sbc = 0
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

        if (r'us-gaap_NetCashProvidedByUsedInOperatingActivities' in str(row.ElementName) or
            r'us-gaap_NetCashProvidedByUsedInOperatingActivitiesContinuingOperations' in str(row.ElementName)
            ):
            cfo_calc = xml_re(str(cells[colm].RoundedNumericAmount))
            if cfo_calc != '---':
                cfo = round(check_neg(str(cells[colm].RoundedNumericAmount), cfo_calc, 'xml') * (dollar_multiplier / 1_000_000), 2) 

        elif (r'us-gaap_PaymentsToAcquirePropertyPlantAndEquipment' in str(row.ElementName) or
              r'us-gaap_PaymentsToAcquireProductiveAssets' in str(row.ElementName)
              ):
            capex_calc = xml_re(str(cells[colm].RoundedNumericAmount))
            if capex_calc != '---':
                capex = round(capex_calc * (dollar_multiplier / 1_000_000), 2)  

        elif (r'us-gaap_PaymentsForRepurchaseOfCommonStock' in str(row.ElementName) or
              r'us-gaap_PaymentsForRepurchaseOfEquity' in str(row.ElementName)
              ):
            buyback_calc = xml_re(str(cells[colm].RoundedNumericAmount))
            if buyback_calc != '---':
                buyback += round(buyback_calc * (dollar_multiplier / 1_000_000), 2) 

        elif (r'us-gaap_ProceedsFromStockOptionsExercised' in str(row.ElementName) or
              r'us-gaap_ProceedsFromIssuanceOfCommonStock' in str(row.ElementName) or
              r'cost_ProceedsFromStockbasedAwardsNet' in str(row.ElementName) or
              r'us-gaap_ProceedsFromIssuanceOrSaleOfEquity' in str(row.ElementName)
              ):
            share_issue_calc = xml_re(str(cells[colm].RoundedNumericAmount))
            if share_issue_calc != '---':
                share_issue += round(share_issue_calc * (dollar_multiplier / 1_000_000), 2) 

        elif (r'us-gaap_PaymentsOfDividendsCommonStock' in str(row.ElementName) and divpaid == 0 or
              r'us-gaap_PaymentsOfDividends' in str(row.ElementName) and divpaid == 0 or
              r'frt_DividendsPaidToCommonAndPreferredShareholders' in str(row.ElementName) and divpaid == 0 or
              r'us-gaap_PaymentsOfOrdinaryDividends<' in str(row.ElementName) and divpaid == 0
              ):
            divpaid_calc = xml_re(str(cells[colm].RoundedNumericAmount))
            if divpaid_calc != '---':
                divpaid += round(divpaid_calc * (dollar_multiplier / 1_000_000), 2)  

        elif r'us-gaap_ShareBasedCompensation' in str(row.ElementName):
            sbc_calc = xml_re(str(cells[colm].RoundedNumericAmount))
            if sbc_calc != '---':
                sbc += round(sbc_calc * (dollar_multiplier / 1_000_000), 2) 
    
    # Calculate Free Cash Flow
    fcf = round(cfo - capex, 2)

    # Net out share issuance
    buyback = round(buyback - share_issue, 2)

    return fcf, buyback, divpaid, sbc

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
    colm = column_finder_annual_xml(soup)
    name = soup.find_all('ReportLongName')
    
    # Initial value
    div = '---'

    # Pull div if reported by quarter
    if ('Unaudited Quarterly Data (Details)' in str(name) or
        'Quarterly Results (unaudited) - (Quarterly Results) (Details)' in str(name)
        ):
        rows = soup.find_all('Row')
        for row in rows:
            
            if (r'>us-gaap_CommonStockDividendsPerShareDeclared' in str(row.ElementName) or
                r'us-gaap_CommonStockDividendsPerShareCashPaid' in str(row.ElementName) or
                r'us-gaap_DividendsPayableAmountPerShare' in str(row.ElementName)
                ):
                cells = row.find_all('Cell')
                obj = re.findall(r'(?:<RoundedNumericAmount>-?)(.*?)(?:</RoundedNumericAmount>)', str(cells), re.M)
                
                # Account for empty cell at start
                if obj[0] == '0':
                    start = 1
                    end = 5
                else:
                    start = 0
                    end = 4

                div = round(sum(map(float, obj[start:end])), 3)
                if div != None and div != 0:
                    return div

    # Go to cell with 12 month data and pull div
    rows = soup.find_all('Row')
    for row in rows:

        if (r'>us-gaap_CommonStockDividendsPerShareDeclared' in str(row.ElementName) or
            r'us-gaap_CommonStockDividendsPerShareCashPaid' in str(row.ElementName)  or
            r'us-gaap_DividendsPayableAmountPerShare' in str(row.ElementName)
            ):
            cells = row.find_all('Cell')
            div_calc = xml_re(str(cells[colm].RoundedNumericAmount))
            if div_calc != None and div_calc != 0:
                div = div_calc
            break
    
    # Check if reporting quarterly div
    if 'QUARTERLY' in str(row).upper() and div != '---':
        div *= 4

    return div


def eps_catch_xml(catch_url, headers):
    ''' Parse EPS and div info if not listed elsewhere

    Args:
        URL of statement
        User agent for SEC
    Returns:
        EPS (float)
        Dividend (float)
    '''

    # Initial value
    eps = div = shares = '---'

    # Get data from site
    content = requests.get(catch_url, headers=headers).content
    soup = BeautifulSoup(content, 'xml')
    colm = column_finder_annual_xml(soup)
    head = soup.RoundingOption
    dollar_multiplier, share_multiplier = multiple_extractor(str(head), shares=True, xml=True)

    # Find table
    rows = soup.find_all('Row')
    for row in rows:
        if r'us-gaap_EarningsPerShareTextBlock' in str(row.ElementName):
            cells = row.find_all('Cell')
            
            # Find eps
            obj = re.search(r'(?:Diluted)(?:.|\n)*?(?:\$)(\d?\d.\d\d)', str(cells[0]))
            if obj is not None:
                eps = float(obj.group(1))

            # Find div
            obj = re.search(r'(?:Less Dividends)(?:.|\n)*?(\d?\d.\d\d)', str(cells[0]))
            if obj is not None:
                div = float(obj.group(1))
        elif r'us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding' in str(row.ElementName):
            cells = row.find_all('Cell')
            share_calc = xml_re(str(cells[colm].RoundedNumericAmount))
            if share_calc != '---':
                shares = round(share_calc * (share_multiplier / 1_000), 2) 

    return eps, div, shares

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

def share_catch_xml(catch_url, headers, per):
    ''' Parse share data if not found in income statement for xml docs

    Args:
        URL of statement
        User agent for SEC
        Fiscal period end date
    Returns:
        Shares (int)     
    '''

    # Get data from site
    content = requests.get(catch_url, headers=headers).content
    soup = BeautifulSoup(content, 'xml')
    
    # Initial values
    shares = '---'
    shares_issued = shares_repurchased = share_sum = 0
    multiple_classes = False

    # Find which column has 12 month data
    colm = column_finder_annual_xml(soup)

    if colm == 0 and 'Class A Common Stock' in str(soup) and 'Class A Special Common Stock' in str(soup) and 'Class B Common Stock' in str(soup):
        multiple_classes = True
        new_per = datetime.strptime(per.replace('.', ''), '%b %d, %Y')
        new_per_str = new_per.strftime('%m/%d/%Y')
        colms = []
        columns = soup.find_all('Column')
        for index, column in enumerate(columns):
            if 'Common Stock' in str(column) and new_per_str in str(column):
                colms.append(index)

    # Determine multiplier
    head = soup.RoundingOption
    dollar_multiplier, share_multiplier = multiple_extractor(str(head), shares=True, xml=True)

    # For Report where shares outstanding are reported as shares issued (PRU)
    # Find new colm number
    if 'Equity (Common Stock and Class B Stock Changes in Number of Shares Issued, Held in Treasury and Outstanding) (Details)' in str(soup.ReportLongName):
        columns = soup.find_all('Column')
        for column in columns:
            if per[-4:] in str(column.MCU.KeyName) and 'Common Stock Outstanding [Member]' in str(column.MCU.KeyName):
                colm = int(str(column.Id)[4:-5]) - 1        
        
        # Find relevant row
        rows = soup.find_all('Row')
        for row in rows:
            cells = row.find_all('Cell')
            if r'us-gaap_CommonStockSharesIssued' in str(row.ElementName):
                shares = round(xml_re(str(cells[colm].RoundedNumericAmount)) * (share_multiplier / 1_000), 2)

        return shares

    # For normal reports
    else:
        # Loop through rows, search for row of interest
        rows = soup.find_all('Row')
        for row in rows:
            cells = row.find_all('Cell')

            if (r'us-gaap_CommonStockSharesOutstanding' in str(row.ElementName)
                ):
                if multiple_classes == True:
                    for colm in colms:
                        share_sum += round(xml_re(str(cells[colm].RoundedNumericAmount)) * (share_multiplier / 1_000), 2)
                    shares = share_sum
                else:
                    shares = round(xml_re(str(cells[colm].RoundedNumericAmount)) * (share_multiplier / 1_000), 2)
                break

            elif (r"us-gaap_CommonStockSharesIssued" in str(row.ElementName)
                ):
                shares_issued = xml_re(str(cells[colm].RoundedNumericAmount))

            elif (r"us-gaap_TreasuryStockShares" in str(row.ElementName)
                ):
                shares_repurchased = xml_re(str(cells[colm].RoundedNumericAmount))

        if shares == '---' and shares_issued != '---' and shares_repurchased != '---':
            shares = round((shares_issued - shares_repurchased) * (share_multiplier / 1_000), 2)

    return shares