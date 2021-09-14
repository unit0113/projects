import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import date
from datetime import datetime
import pandas as pd
import math
import yfinance as yf
import numpy as np


# Ticker to lookup
stock = yf.Ticker('AAPL')

# Summary Data
info = stock.info
sector = info['sector']
sub_sector = info['industry']
business_description = info['longBusinessSummary']
price_target_low = info['targetLowPrice']
price_target_median = info['targetMedianPrice']
price_target_high = info['targetHighPrice']
analyst_rec = info['recommendationMean']
short_name = info['shortName']
yr_change = info['52WeekChange']
short_percentage = info['shortPercentOfFloat']
beta = info['beta']
peg_ratio = info['pegRatio']
logo_url = info['logo_url']
current_price = info['currentPrice']
analyst_upside = float("{:.4f}".format((price_target_median - current_price) / current_price))
year_high = info['fiftyTwoWeekHigh']
year_low = info['fiftyTwoWeekLow']
avg_yield = info['fiveYearAvgDividendYield']


# Dividends
dividends = stock.dividends
ttm_div_growth = (sum(list(dividends[-4:])) / sum(list(dividends[-8:-4]))) - 1.0

# Stock splits
splits = stock.splits

# Historical market data
chart_data = stock.history(period="10y", interval='1wk')['Close']
ten_yr_change = float("{:.4f}".format((chart_data[-1] - chart_data[0]) / chart_data[0]))
mid_point = len(chart_data) // 2
five_yr_change = float("{:.4f}".format((chart_data[-1] - chart_data[mid_point]) / chart_data[mid_point]))




per =  [datetime(2020, 9, 26, 0, 0), datetime(2019, 9, 28, 0, 0), datetime(2018, 9, 29, 0, 0), datetime(2017, 9, 30, 0, 0), datetime(2016, 9, 24, 0, 0), datetime(2015, 9, 26, 0, 0), datetime(2014, 9, 27, 0, 0), datetime(2013, 9, 28, 0, 0), datetime(2012, 9, 29, 0, 0), datetime(2011, 9, 24, 0, 0), datetime(2010, 9, 25, 0, 0)]
eps = [3.28, 11.89, 11.91, 9.21, 8.31, 9.22, 6.45, 39.75, 44.15, 27.68, 15.15]
div = [0.68, 3.0, 2.72, 2.4, 2.18, 1.98, 1.82, 11.4, 2.65, '---', '---']
shares = [17528214, 4648913, 5000109, 5251692, 5500281, 5793069, 6122663, 931662, 945355, 936645, 924712]

# split checker
def split_calculator(per, splits, eps, div, shares):
    ''' Adjust per share metrics based on stock splits

    Args:
        list of report end dates (datetime object)

    Returns:
        dividend (float)
    '''

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
                split_index += 1
            else:
                break
        
        # Create list of split factors
        for i in range(1, len(per)):
            if split_list_date[split_index] < per[i-1] and split_list_date[split_index] > per[i]:
                split_factor *= split_list_splits[split_index]
                split_index += 1
            split_return.append(split_factor)

        
        # If end of relevant splits

    # Adjust div values in case of empty
    div = [0 if elem == '---' else elem for elem in div]

    # Adjust values based on split factor
    eps = np.divide(eps, split_return)
    eps = list(np.around(np.array(eps),2))
    div = np.divide(div, split_return)
    div = list(np.around(np.array(div),2))
    adjusted_shares = np.divide(shares, split_return)
    adjusted_shares = list(np.around(np.array(adjusted_shares),2))

    # Unadjust div values in case of empty
    div = ['---' if elem == 0 else elem for elem in div]

    return eps, div, adjusted_shares


split_calculator(per, splits, eps, div, shares)