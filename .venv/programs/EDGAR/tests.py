import final_answers as ans
import test_html_new as htest
import test_xml_new as xtest
import statement_parser as sp
import statement_tests as st
import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import date
from datetime import datetime
import pandas as pd
import timeit


# Header data for data pull
with open(r'C:\Users\unit0\OneDrive\Desktop\EDGAR\user_agent.txt') as f:
    data = f.read()
    headers = json.loads(data)

fails = 0

# Start timer
starttime = timeit.default_timer()

def rev_test():
    global fails
    # HTML test
    for i, url in enumerate(st.rev_url_list_htm):
        result = list(sp.rev_htm(url, headers, st.get_sum_per(url)))
        answer = st.rev_answers_htm[i]
        print(result)
        print(answer)
        if result != answer:
            print(url)
            fails += 1
        print('-'*100)
    
    # XML Test
    for i, url in enumerate(st.rev_url_list_xml):
        result = list(sp.rev_xml(url, headers))
        answer = st.rev_answers_xml[i]
        print(result)
        print(answer)
        if result != answer:
            print(url)
            fails += 1
        print('-'*100)

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

def bs_test():
    global fails
    # HTML test
    for i, url in enumerate(st.bs_url_list_htm):
        result = list(sp.bs_htm(url, headers))
        answer = st.bs_answers_htm[i]
        print(result)
        print(answer)
        if result != answer:
            print(url)
            fails += 1
        print('-'*100)
    
    # XML Test
    for i, url in enumerate(st.bs_url_list_xml):
        result = list(sp.bs_xml(url, headers))
        answer = st.bs_answers_xml[i]
        print(result)
        print(answer)
        if result != answer:
            print(url)
            fails += 1
        print('-'*100)

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

def cf_test():
    global fails
    # HTML test
    for i, url in enumerate(st.cf_url_list_htm):
        result = list(sp.cf_htm(url, headers))
        answer = st.cf_answers_htm[i]
        print(result)
        print(answer)
        if result != answer:
            print(url)
            fails += 1
        print('-'*100)
    
    # XML Test
    for i, url in enumerate(st.cf_url_list_xml):
        result = list(sp.cf_xml(url, headers))
        answer = st.cf_answers_xml[i]
        print(result)
        print(answer)
        if result != answer:
            print(url)
            fails += 1
        print('-'*100)

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

def div_test():
    global fails
    # HTML test
    for i, url in enumerate(st.div_url_list_htm):
        result = sp.div_htm(url, headers)
        answer = st.div_answers_htm[i]
        print(result)
        print(answer)
        if result != answer:
            print(url)
            fails += 1
        print('-'*100)
    
    # XML Test
    for i, url in enumerate(st.div_url_list_xml):
        result = sp.div_xml(url, headers)
        answer = st.div_answers_xml[i]
        print(result)
        print(answer)
        if result != answer:
            print(url)
            fails += 1
        print('-'*100)

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

def eps_catch_test():
    global fails
    # HTML test
    for i, url in enumerate(st.catch_url_list_htm):
        result = list(sp.eps_catch_htm(url, headers, st.eps_list_htm[i]))
        answer = st.catch_answers_htm[i]
        print(result)
        print(answer)
        if result != answer:
            print(url)
            fails += 1
        print('-'*100)

    # XML test
    for i, url in enumerate(st.catch_url_list_xml):
        result = list(sp.eps_catch_xml(url, headers, st.eps_list_xml[i]))
        answer = st.catch_answers_xml[i]
        print(result)
        print(answer)
        if result != answer:
            print(url)
            fails += 1
        print('-'*100)

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# TESTS!!!!!
rev_test()
#bs_test()
#cf_test()
#div_test()
#eps_catch_test()

# Calculate duration of tests
delta_time = timeit.default_timer() - starttime

# Print number of failures
if fails == 1:
    print("There was one failure")
elif fails > 1:
    print(f"There were {fails} failures")
else:
    print("Tests successful. Huzzah")

# Print time
print(f'Duration of test: {delta_time} seconds')