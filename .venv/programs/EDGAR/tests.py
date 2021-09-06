import final_answers as ans
import test_html_new as htest
import test_xml_new as xtest
import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import date
from datetime import datetime
import pandas as pd


fails = 0

def rev_test():
    global fails
    # HTML test
    for i, url in enumerate(htest.rev_url_list):
        result = list(htest.rev_htm_test(url))
        answer = htest.rev_answers[i]
        print(result)
        print(answer)
        if result != answer:
            print(url)
            fails += 1
        print('-'*100)
    
    # XML Test
    for i, url in enumerate(xtest.rev_url_list):
        result = list(xtest.rev_xml_test(url))
        answer = xtest.rev_answers[i]
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
    for i, url in enumerate(htest.bs_url_list):
        result = list(htest.bs_htm_test(url))
        answer = htest.bs_answers[i]
        print(result)
        print(answer)
        if result != answer:
            print(url)
            fails += 1
        print('-'*100)
    
    # XML Test
    for i, url in enumerate(xtest.bs_url_list):
        result = list(xtest.bs_xml_test(url))
        answer = xtest.bs_answers[i]
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
    for i, url in enumerate(htest.cf_url_list):
        result = list(htest.cf_htm_test(url))
        answer = htest.cf_answers[i]
        print(result)
        print(answer)
        if result != answer:
            print(url)
            fails += 1
        print('-'*100)
    
    # XML Test
    for i, url in enumerate(xtest.cf_url_list):
        result = list(xtest.cf_xml_test(url))
        answer = xtest.cf_answers[i]
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
    for i, url in enumerate(htest.div_url_list):
        result = htest.div_htm_test(url)
        answer = htest.div_answers[i]
        print(result)
        print(answer)
        if result != answer:
            print(url)
            fails += 1
        print('-'*100)
    
    # XML Test
    for i, url in enumerate(xtest.div_url_list):
        result = xtest.div_xml_test(url)
        answer = xtest.div_answers[i]
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

# Print number of failures
if fails == 1:
    print("There was one failure")
elif fails > 1:
    print(f"There were {fails} failures")
else:
    print("Tests successful. Huzzah")