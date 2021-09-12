import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import date
from datetime import datetime
import pandas as pd
import math
from selenium import webdriver
import string
import csv
from io import StringIO
import yfinance as yf


# Ticker to lookup
stock = yf.Ticker('MSFT')

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
print(analyst_upside)
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

