import json
import statement_parser as sp
import edgarquery as eq
import math
import re
import yfinance as yf
import requests
from bs4 import BeautifulSoup

# Header data for data pull
try:
    with open(r'user_agent.txt') as f:
        data = f.read()
        headers = json.loads(data)
except:
    with open(r'C:\Users\unit0\Desktop\EDGAR\user_agent.txt') as f:
        data = f.read()
        headers = json.loads(data)

sum_url = r'https://www.sec.gov/Archives/edgar/data/1164727/000155837019000806/R1.htm'

# Sum Test
#print(sp.sum_htm(sum_url, headers))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

def get_sum_per(url):
    sum_url = url[:-6] + 'R1.htm'
    fy, per, name = sp.sum_htm(sum_url, headers)
    return per

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# rev, gross, research, oi, net, eps, share_sum, div, ffo
rev_url_list_htm = [r'https://www.sec.gov/Archives/edgar/data/789019/000156459021039151/R2.htm', r'https://www.sec.gov/Archives/edgar/data/320193/000032019320000096/R2.htm', r'https://www.sec.gov/Archives/edgar/data/320193/000162828016020309/R2.htm', 
                    r'https://www.sec.gov/Archives/edgar/data/320193/000119312513416534/R2.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000156459017014900/R2.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000119312513310206/R2.htm',
                    r'https://www.sec.gov/Archives/edgar/data/1018724/000101872421000004/R3.htm', r'https://www.sec.gov/Archives/edgar/data/1018724/000101872418000005/R3.htm', r'https://www.sec.gov/Archives/edgar/data/1018724/000101872414000006/R3.htm',
                    r'https://www.sec.gov/Archives/edgar/data/1652044/000165204421000010/R4.htm', r'https://www.sec.gov/Archives/edgar/data/1652044/000165204419000004/R4.htm', r'https://www.sec.gov/Archives/edgar/data/1652044/000165204416000012/R4.htm',
                    r'https://www.sec.gov/Archives/edgar/data/1403161/000140316120000070/R4.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000140316117000044/R4.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000140316114000017/R4.htm',
                    r'https://www.sec.gov/Archives/edgar/data/789019/000156459019027952/R2.htm', r'https://www.sec.gov/Archives/edgar/data/320193/000119312512444068/R2.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000119312515272806/R2.htm',
                    r'https://www.sec.gov/Archives/edgar/data/320193/000119312514383437/R2.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000119312512316848/R2.htm', r'https://www.sec.gov/Archives/edgar/data/320193/000032019318000145/R2.htm', 
                    r'https://www.sec.gov/Archives/edgar/data/1652044/000165204418000007/R4.htm', r'https://www.sec.gov/Archives/edgar/data/354950/000035495015000008/R4.htm', r'https://www.sec.gov/Archives/edgar/data/1318605/000156459017003118/R4.htm',
                    r'https://www.sec.gov/Archives/edgar/data/1318605/000119312514069681/R4.htm', r'https://www.sec.gov/Archives/edgar/data/909832/000090983220000017/R2.htm', r'https://www.sec.gov/Archives/edgar/data/909832/000144530513002422/R4.htm',
                    r'https://www.sec.gov/Archives/edgar/data/909832/000119312512428890/R4.htm', r'https://www.sec.gov/Archives/edgar/data/1538990/000155837021002003/R4.htm', r'https://www.sec.gov/Archives/edgar/data/764180/000076418020000018/R4.htm',
                    r'https://www.sec.gov/Archives/edgar/data/916365/000091636519000035/R2.htm', r'https://www.sec.gov/Archives/edgar/data/1551152/000104746916010239/R2.htm', r'https://www.sec.gov/Archives/edgar/data/318154/000031815420000017/R2.htm',
                    r'https://www.sec.gov/Archives/edgar/data/318154/000119312512086670/R3.htm', r'https://www.sec.gov/Archives/edgar/data/1287865/000119312518067584/R4.htm', r'https://www.sec.gov/Archives/edgar/data/1287865/000156459021009901/R4.htm',
                    r'https://www.sec.gov/Archives/edgar/data/1553023/000155302321000012/R4.htm', r'https://www.sec.gov/Archives/edgar/data/1553023/000155302318000029/R4.htm', r'https://www.sec.gov/Archives/edgar/data/1553023/000155302316000152/R4.htm',
                    r'https://www.sec.gov/Archives/edgar/data/764478/000076447813000014/R4.htm', r'https://www.sec.gov/Archives/edgar/data/1253986/000110465921025551/R4.htm', r'https://www.sec.gov/Archives/edgar/data/753308/000075330821000014/R2.htm',
                    r'https://www.sec.gov/Archives/edgar/data/753308/000075330817000060/R2.htm', r'https://www.sec.gov/Archives/edgar/data/202058/000020205821000008/R2.htm', r'https://www.sec.gov/Archives/edgar/data/732717/000073271717000021/R2.htm',
                    r'https://www.sec.gov/Archives/edgar/data/1108524/000110852418000011/R4.htm', r'https://www.sec.gov/Archives/edgar/data/97210/000119312519059974/R4.htm', r'https://www.sec.gov/Archives/edgar/data/97745/000009774521000011/R4.htm',
                    r'https://www.sec.gov/Archives/edgar/data/21076/000002107621000016/R2.htm', r'https://www.sec.gov/Archives/edgar/data/886982/000119312521049380/R2.htm', r'https://www.sec.gov/Archives/edgar/data/51143/000104746918001117/R2.htm',
                    r'https://www.sec.gov/Archives/edgar/data/1164727/000119312514061205/R2.htm', r'https://www.sec.gov/Archives/edgar/data/751364/000075136421000018/R4.htm', r'https://www.sec.gov/Archives/edgar/data/751364/000075136419000024/R4.htm',
                    r'https://www.sec.gov/Archives/edgar/data/1556593/000155659320000004/R4.htm', r'https://www.sec.gov/Archives/edgar/data/726728/000072672821000043/R4.htm', r'https://www.sec.gov/Archives/edgar/data/1137774/000119312515056366/R4.htm']

rev_answers_htm = [[168088.0, 115856.0, 20716.0, 69916.0, 61271.0, 8.05, 7608000.0, '---', 61271.0], [274515.0, 104956.0, 18752.0, 66288.0, 57411.0, 3.28, 17528214.0, '---', 57411.0], [215639.0, 84263.0, 10045.0, 60024.0, 45687.0, 8.31, 5500281.0, 2.18, 45687.0],
                   [170910.0, 64304.0, 4475.0, 48999.0, 37037.0, 39.75, 931662.0, 11.4, 37037.0], [89950.0, 55689.0, 13037.0, 22326.0, 21204.0, 2.71, 7832000.0, 1.56, 21204.0], [77849.0, 57600.0, 10411.0, 26764.0, 21863.0, 2.58, 8470000.0, 0.92, 21863.0],
                   [386064.0, 152757.0, 42740.0, 22899.0, 21331.0, 41.83, 510000.0, '---', 21331.0], [177866.0, 65932.0, 22620.0, 4106.0, 3033.0, 6.15, 493000.0, '---', 3033.0], [74452.0, 20271.0, 6565.0, 745.0, 274.0, 0.59, 465000.0, '---', 274.0],
                   [182527.0, 97795.0, 27573.0, 41224.0, 40269.0, 58.61, '---', '---', 40269.0], [136819.0, 77270.0, 21419.0, 26321.0, 30736.0, 43.7, '---', '---', 30736.0], [74989.0, 46825.0, 12282.0, 19360.0, 15826.0, 22.84, '---', '---', 15826.0],
                   [21846.0, 19729.0, 0, 14081.0, 10866.0, 4.89, 2223000.0, '---', 11633.0], [18358.0, 16388.0, 0, 12144.0, 6699.0, 2.8, 2395000.0, '---', 7255.0], [12702.0, 10514.0, 0, 7697.0, 5438.0, 8.62, 631000.0, '---', 5873.0],
                   [125843.0, 82933.0, 16876.0, 42959.0, 39240.0, 5.06, 7753000.0, '---', 39240.0], [156508.0, 68662.0, 3381.0, 55241.0, 41733.0, 44.15, 945355.0, 2.65, 41733.0], [93580.0, 60542.0, 12046.0, 18161.0, 12193.0, 1.48, 8254000.0, 1.24, 12193.0],
                   [182795.0, 70537.0, 6041.0, 52503.0, 39510.0, 6.45, 6122663.0, 1.82, 39510.0], [73723.0, 56193.0, 9811.0, 21763.0, 16978.0, 2.0, 8506000.0, 0.8, 16978.0], [265595.0, 101839.0, 14236.0, 70898.0, 59531.0, 11.91, 5000109.0, '---', 59531.0],
                   [110855.0, 65272.0, 16625.0, 26146.0, 12662.0, 18.0, '---', '---', 12662.0], [83176.0, 28954.0, 0, 10469.0, 6345.0, 4.71, 1346000.0, '---', 7996.0], [7000.13, 1599.26, 834.41, -667.34, -674.91, -4.68, 144212.0, '---', -674.91],
                   [2013.5, 456.26, 231.98, -61.28, -74.01, -0.62, 119421.41, '---', -74.01], [166761.0, 21822.0, 0, 5435.0, 4002.0, 9.02, 443901.0, '---', 4002.0], [105156.0, 13208.0, 0, 3053.0, 2039.0, 4.63, 440512.0, 8.17, 2039.0],
                   [99137.0, 12314.0, 0, 2759.0, 1709.0, 3.89, 439373.0, 1.03, 1709.0], [694.27, 502.53, 0, 186.93, 212.61, 0.84, 252651.04, '---', 432.77], [25110.0, 12711.0, 0, 10326.0, -1293.0, -0.7, '---', '---', -1293.0],
                   [7911.05, 2702.53, 0, 701.74, 532.36, 4.31, 123471.0, 1.2, 709.71], [22859.0, 18359.0, 4285.0, 7537.0, 5144.0, 3.13, 1637000.0, 2.1, 5144.0], [23362.0, 19006.0, 4116.0, 9674.0, 7842.0, 12.88, 609000.0, '---', 7842.0],
                   [15582.0, 13155.0, 3167.0, 4312.0, 3683.0, 4.04, 912000.0, '---', 3683.0], [704.75, 698.94, 0, 485.58, 289.79, 0.82, 350441.0, '---', 407.47], [1249.24, 920.51, 0, 499.71, 431.45, 0.81, 530461.0, '---', 698.53],
                   [1033.5, 621.9, 0, 40.1, 41.4, 0.35, 117600.0, '---', 490.8], [672.0, 436.9, 0, 24.1, -83.5, -0.95, 88900.0, '---', 175.4], [399.3, 224.4, 0, 22.8, -15.4, -0.3, 54300.0, '---', 126.1],
                   [45085.0, 10649.0, 0, -125.0, -441.0, -1.3, 338600.0, '---', -441.0], [603.73, 434.51, 0, 333.53, 163.4, 1.41, 133969.3, '---', 171.42], [17997.0, 14458.0, 0, 5116.0, 2919.0, 1.48, '---', '---', 2919.0],
                   [16155.0, 12113.0, 0, 4608.0, 2912.0, 6.25, 465800.0, '---', 2912.0], [18194.0, 5308.0, 0, 1226.0, 1119.0, 5.19, '---', '---', 1119.0], [163786.0, 86541.0, 0, 24347.0, 12976.0, 2.1, '---', '---', 38823.0],
                   [10480.01, 7706.49, 1553.07, 235.77, 127.48, 0.17, 734598.0, '---', 127.48], [2100.8, 1220.39, 301.5, 473.8, 451.78, 2.35, 192605.0, 0.36, 451.78], [32218.0, 16004.0, 1181.0, 7794.0, 6375.0, 15.96, 399000.0, '---', 6375.0],
                   [7341.0, 3199.0, 149.0, 927.0, 710.0, 5.58, 127299.0, '---', 710.0], [44560.0, 35622.0, 0, 12479.0, 8915.0, 24.74, 360300.0, '---', 10817.0], [79139.0, 36227.0, 5787.0, 10332.0, 5753.0, 6.14, 937385.62, '---', 5753.0],
                   [8322.0, 3136.0, 222.0, -3631.0, -2462.0, -4.94, '---', 1.225, -1100.0], [660.68, 593.03, 0, 374.49, 210.86, 1.22, 172217.08, '---', 428.68], [622.66, 597.56, 0, 424.76, 258.12, 1.65, 156295.62, '---', 460.73],
                   [2585.64, 1651.89, 0, 1606.62, 550.01, 1.34, 408990.11, 2.0, 585.59], [1651.62, 1237.68, 0, 325.5, 394.47, 1.14, 345415.26, '---', 1218.74], [54105.0, 13566.0, 0, 1759.0, 1381.0, 3.23, '---', 2.17, 1381.0]]

rev_url = r'https://www.sec.gov/Archives/edgar/data/1618921/000114036115038791/R6.htm'

# Rev Test
print(sp.rev_htm(rev_url, headers, get_sum_per(rev_url)))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# cash, cur_assets, assets, debt, cur_liabilities, liabilities, equity
bs_url_list_htm = [r'https://www.sec.gov/Archives/edgar/data/1403161/000140316120000070/R2.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000140316118000055/R2.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000140316116000058/R2.htm',
                   r'https://www.sec.gov/Archives/edgar/data/1403161/000140316114000017/R2.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000119312511315956/R2.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000156459021039151/R4.htm', 
                   r'https://www.sec.gov/Archives/edgar/data/789019/000156459019027952/R4.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000156459017014900/R5.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000119312515272806/R5.htm',
                   r'https://www.sec.gov/Archives/edgar/data/789019/000119312513310206/R5.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000119312512316848/R3.htm', r'https://www.sec.gov/Archives/edgar/data/320193/000032019320000096/R4.htm',
                   r'https://www.sec.gov/Archives/edgar/data/320193/000032019317000070/R5.htm', r'https://www.sec.gov/Archives/edgar/data/320193/000119312514383437/R5.htm', r'https://www.sec.gov/Archives/edgar/data/320193/000119312511282113/R3.htm',
                   r'https://www.sec.gov/Archives/edgar/data/1652044/000165204421000010/R2.htm', r'https://www.sec.gov/Archives/edgar/data/1652044/000165204418000007/R2.htm', r'https://www.sec.gov/Archives/edgar/data/1652044/000165204416000012/R2.htm',
                   r'https://www.sec.gov/Archives/edgar/data/354950/000035495017000005/R2.htm', r'https://www.sec.gov/Archives/edgar/data/354950/000035495020000015/R2.htm', r'https://www.sec.gov/Archives/edgar/data/909832/000090983220000017/R4.htm',
                   r'https://www.sec.gov/Archives/edgar/data/909832/000090983218000013/R2.htm', r'https://www.sec.gov/Archives/edgar/data/77476/000007747621000007/R5.htm', r'https://www.sec.gov/Archives/edgar/data/916365/000091636520000050/R5.htm',
                   r'https://www.sec.gov/Archives/edgar/data/916365/000091636516000140/R3.htm', r'https://www.sec.gov/Archives/edgar/data/319201/000031920120000047/R2.htm', r'https://www.sec.gov/Archives/edgar/data/318154/000031815414000004/R5.htm',
                   r'https://www.sec.gov/Archives/edgar/data/1538990/000155837021002003/R2.htm', r'https://www.sec.gov/Archives/edgar/data/1053507/000105350721000026/R2.htm', r'https://www.sec.gov/Archives/edgar/data/1287865/000119312517065943/R2.htm',
                   r'https://www.sec.gov/Archives/edgar/data/1045609/000119312515062622/R2.htm', r'https://www.sec.gov/Archives/edgar/data/1253986/000110465921025551/R2.htm', r'https://www.sec.gov/Archives/edgar/data/753308/000075330814000025/R5.htm',
                   r'https://www.sec.gov/Archives/edgar/data/753308/000075330813000023/R5.htm', r'https://www.sec.gov/Archives/edgar/data/2969/000000296919000051/R5.htm', r'https://www.sec.gov/Archives/edgar/data/1561550/000156459021009770/R2.htm',
                   r'https://www.sec.gov/Archives/edgar/data/1099590/000156276221000079/R2.htm', r'https://www.sec.gov/Archives/edgar/data/18230/000001823021000063/R4.htm',r'https://www.sec.gov/Archives/edgar/data/18230/000001823020000056/R5.htm',
                   r'https://www.sec.gov/Archives/edgar/data/1574540/000110465914021828/R2.htm', r'https://www.sec.gov/Archives/edgar/data/886982/000119312521049380/R4.htm', r'https://www.sec.gov/Archives/edgar/data/886982/000119312513085474/R4.htm',
                   r'https://www.sec.gov/Archives/edgar/data/886982/000119312518056383/R4.htm', r'https://www.sec.gov/Archives/edgar/data/895421/000089542120000265/R4.htm', r'https://www.sec.gov/Archives/edgar/data/895421/000119312515071980/R2.htm',
                   r'https://www.sec.gov/Archives/edgar/data/751364/000075136420000021/R2.htm', r'https://www.sec.gov/Archives/edgar/data/751364/000075136413000007/R2.htm', r'https://www.sec.gov/Archives/edgar/data/888491/000088849121000006/R2.htm',
                   r'https://www.sec.gov/Archives/edgar/data/1137774/000113777421000049/R2.htm', r'https://www.sec.gov/Archives/edgar/data/1393311/000139331117000008/R2.htm', r'https://www.sec.gov/Archives/edgar/data/1298675/000129867521000006/R2.htm',
                   r'https://www.sec.gov/Archives/edgar/data/92122/000009212220000017/R8.htm', r'https://www.sec.gov/Archives/edgar/data/1063761/000155837020001135/R2.htm', r'https://www.sec.gov/Archives/edgar/data/914208/000091420821000331/R2.htm',
                   r'https://www.sec.gov/Archives/edgar/data/86312/000008631221000011/R4.htm']

bs_answers_htm = [[16289.0, 27645.0, 37201.0, 21071.0, 14510.0, 44709.0, 36210.0], [8162.0, 18216.0, 26473.0, 16630.0, 11305.0, 35219.0, 34006.0], [5619.0, 14313.0, 21735.0, 15882.0, 8046.0, 31123.0, 32912.0],
                  [1971.0, 9562.0, 15405.0, 0, 6006.0, 11156.0, 27413.0], [2127.0, 9190.0, 11656.0, 0, 3451.0, 8323.0, 26437.0], [130334.0, 184406.0, 276268.0, 50074.0, 88657.0, 191791.0, 141988.0],
                  [133819.0, 175552.0, 236780.0, 66662.0, 69420.0, 184226.0, 102330.0], [132981.0, 159851.0, 195858.0, 76073.0, 64527.0, 168692.0, 72394.0], [96526.0, 124712.0, 154449.0, 27808.0, 49858.0, 96140.0, 80083.0],
                  [77022.0, 101466.0, 124693.0, 12601.0, 37417.0, 63487.0, 78944.0], [63040.0, 85084.0, 104649.0, 10713.0, 32688.0, 54908.0, 66363.0], [38016.0, 143713.0, 323888.0, 98667.0, 105392.0, 258549.0, 65339.0],
                  [20289.0, 128645.0, 367304.0, 97207.0, 100814.0, 241272.0, 134047.0], [13844.0, 68531.0, 223081.0, 28987.0, 63448.0, 120292.0, 111547.0], [9815.0, 44988.0, 111939.0, 0, 27970.0, 39756.0, 76615.0],
                  [136694.0, 174296.0, 296996.0, 13932.0, 56834.0, 97072.0, 222544.0], [101871.0, 124308.0, 177856.0, 3969.0, 24183.0, 44793.0, 152502.0], [73066.0, 90114.0, 127745.0, 3990.0, 19310.0, 27130.0, 120331.0],
                  [2538.0, 17724.0, 40873.0, 22349.0, 14133.0, 38633.0, 4333.0], [2133.0, 19810.0, 48982.0, 28670.0, 18375.0, 54352.0, -3116.0], [12277.0, 28120.0, 55556.0, 7514.0, 24844.0, 36851.0, 18284.0],
                  [6055.0, 20289.0, 40830.0, 6487.0, 19926.0, 27727.0, 12799.0], [8185.0, 23001.0, 54846.0, 40370.0, 23372.0, 79366.0, 13454.0], [84.24, 1787.89, 5164.78, 366.48, 1247.6, 3722.14, 1567.12],
                  [63.81, 1485.43, 2360.57, 150.0, 671.28, 977.53, 1393.29], [1234.41, 4723.55, 5843.15, 3469.67, 1699.79, 6598.95, 2665.42], [3805.0, 27367.0, 37895.0, 29623.0, 7947.0, 44029.0, 22096.0],
                  [166.38, 816.7, 9004.34, 2212.63, 1736.61, 3988.56, 5015.78], [1746.3, 2905.6, 26111.0, 28497.7, 3655.5, 42453.0, 4093.5], [83.24, 1318.2, 6418.54, 2909.34, 207.71, 3165.31, 3248.38],
                  [350.69, 454.14, 25818.22, 9380.2, 628.0, 10634.62, 13975.51], [339.53, 5835.32, 7555.54, 3321.81, 359.67, 6178.3, 1344.37], [438.0, 5842.0, 69306.0, 23969.0, 9189.0, 51266.0, 18040.0],
                  [329.0, 5237.0, 64439.0, 23177.0, 8879.0, 48371.0, 16068.0], [2248.7, 4618.3, 17726.2, 2907.3, 1820.9, 7554.5, 11053.6], [224.93, 1718.08, 1870.61, 575.86, 297.84, 932.85, 957.43],
                  [1856.39, 5346.81, 6426.96, 860.88, 3635.88, 4874.75, 1651.58], [9352.0, 39464.0, 70622.0, 25999.0, 25717.0, 62946.0, 15378.0], [8284.0, 39193.0, 70692.0, 26281.0, 26621.0, 63824.0, 14629.0],
                  [56.48, 57.32, 269.75, 42.82, 9.21, 52.02, 212.3], [155842.0, 643508.0, 1163028.0, 213481.0, 831164.0, 1067096.0, 95932.0], [72669.0, 491921.0, 938555.0, 167305.0, 653139.0, 862839.0, 75716.0],
                  [110051.0, 572442.0, 916776.0, 217687.0, 599924.0, 834533.0, 82243.0], [82171.0, 463227.0, 886179.0, 192627.0, 620105.0, 812732.0, 81549.0], [87591.0, 444441.0, 791763.0, 152772.0, 576634.0, 729406.0, 70900.0],
                  [1.11, 32.88, 7434.87, 2988.36, 18.25, 3103.18, 4331.68], [2.08, 71.52, 3988.03, 1586.96, 17.53, 1690.44, 2296.28], [163.53, 1297.94, 8845.71, 5169.25, 280.82, 5460.84, 3841.88],
                  [13701.0, 13701.0, 940722.0, 19718.0, 483343.0, 872512.0, 67425.], [183.69, 183.69, 9917.62, 390.75, 297.94, 688.68, 9411.91], [3.59, 6.23, 4778.14, 2364.67, 227.44, 2687.88, 1832.22],
                  [1975.0, 9817.0, 112884.0, 41798.0, 12546.0, 86650.0, 27505.0], [669.37, 1501.52, 31231.63, 24163.23, 1390.68, 28101.32, 2526.4], [1408.4, 18377.4, 20282.2, 8796.7, 10574.6, 21483.4, 14361.8],
                  [721.0, 21914.0, 112471.0, 6550.0, 74360.0, 87563.0, 29201.0]]

bs_url = r'https://www.sec.gov/Archives/edgar/data/38777/000003877721000205/R4.htm'

# BS Test
#print(sp.bs_htm(bs_url, headers, get_sum_per(bs_url)))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# fcf, buyback, divpaid, sbc
cf_url_list_htm = [r'https://www.sec.gov/Archives/edgar/data/1652044/000165204418000007/R8.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000140316120000070/R8.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000140316118000055/R8.htm',
                   r'https://www.sec.gov/Archives/edgar/data/1403161/000140316116000058/R8.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000140316114000017/R8.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000138410812000011/R8.htm',
                   r'https://www.sec.gov/Archives/edgar/data/789019/000156459021039151/R6.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000156459019027952/R6.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000156459017014900/R7.htm',
                   r'https://www.sec.gov/Archives/edgar/data/789019/000119312515272806/R7.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000119312513310206/R7.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000119312512316848/R5.htm',
                   r'https://www.sec.gov/Archives/edgar/data/320193/000032019320000096/R7.htm', r'https://www.sec.gov/Archives/edgar/data/320193/000032019317000070/R8.htm', r'https://www.sec.gov/Archives/edgar/data/320193/000119312514383437/R8.htm',
                   r'https://www.sec.gov/Archives/edgar/data/320193/000119312511282113/R6.htm', r'https://www.sec.gov/Archives/edgar/data/1652044/000165204421000010/R8.htm', r'https://www.sec.gov/Archives/edgar/data/1652044/000165204416000012/R8.htm',
                   r'https://www.sec.gov/Archives/edgar/data/789019/000119312511200680/R5.htm', r'https://www.sec.gov/Archives/edgar/data/354950/000035495021000089/R7.htm', r'https://www.sec.gov/Archives/edgar/data/354950/000035495015000008/R8.htm',
                   r'https://www.sec.gov/Archives/edgar/data/1318605/000156459019003165/R8.htm', r'https://www.sec.gov/Archives/edgar/data/1318605/000156459016013195/R8.htm', r'https://www.sec.gov/Archives/edgar/data/909832/000144530513002422/R7.htm',
                   r'https://www.sec.gov/Archives/edgar/data/1538990/000155837021002003/R8.htm', r'https://www.sec.gov/Archives/edgar/data/1538990/000155837017001015/R8.htm', r'https://www.sec.gov/Archives/edgar/data/764180/000076418021000037/R6.htm',
                   r'https://www.sec.gov/Archives/edgar/data/14272/000001427216000288/R6.htm', r'https://www.sec.gov/Archives/edgar/data/1553023/000155302316000152/R8.htm', r'https://www.sec.gov/Archives/edgar/data/1045609/000156459021005312/R8.htm',
                   r'https://www.sec.gov/Archives/edgar/data/1045609/000156459019002872/R7.htm', r'https://www.sec.gov/Archives/edgar/data/97476/000009747614000007/R7.htm', r'https://www.sec.gov/Archives/edgar/data/202058/000020205821000008/R6.htm',
                   r'https://www.sec.gov/Archives/edgar/data/1501585/000119312512138529/R5.htm', r'https://www.sec.gov/Archives/edgar/data/732717/000073271721000012/R7.htm', r'https://www.sec.gov/Archives/edgar/data/783280/000078328020000010/R5.htm',
                   r'https://www.sec.gov/Archives/edgar/data/1326160/000132616021000063/R6.htm']

cf_answers_htm = [[23907.0, 4846.0, 0.0, 7679.0], [9704.0, 7924.0, 2664.0, 416.0], [11995.0, 7028.0, 1918.0, 327.0],
                  [5051.0, 7062.0, 1350.0, 221.0], [6652.0, 4027.0, 1006.0, 172.0], [4633.0, 536.0, 595.0, 147.0],
                  [56118.0, 25692.0, 16521.0, 6118.0], [38260.0, 18401.0, 13811.0, 4652.0], [31378.0, 11016.0, 11845.0, 3266.0],
                  [23136.0, 13809.0, 9882.0, 2574.0], [24576.0, 4429.0, 7455.0, 2406.0], [29321.0, 3116.0, 6385.0, 2244.0],
                  [73365.0, 71478.0, 14081.0, 6829.0], [51147.0, 32345.0, 12769.0, 4840.0], [50142.0, 44270.0, 11126.0, 2863.0],
                  [33269.0, -831.0, 0, 1168.0], [42843.0, 31149.0, 0, 12991.0], [16109.0, 1780.0, 47.0, 5203.0],
                  [24639.0, 9133.0, 5180.0, 2166.0], [16376.0, 465.0, 6451.0, 310.0], [6800.0, 6748.0, 2530.0, 225.0],
                  [-2.92, -295.72, 0, 749.02],[-2159.35, -836.61, 0, 198.0], [1354.0, -16.0, 3560.0, 285.0],
                  [431.59, -690.74, 353.2, 4.67], [246.3, -476.98, 162.62, 7.02], [8154.0, 0.0, 6290.0, 0],
                  [1012.0, -266.0, 2477.0, 235.0], [-492.7, -799.5, 80.8, 14.4], [2937.01, 44.25, 1722.99, 109.83],
                  [1803.56, 6.89, 1123.37, 76.09], [2972.0, 4182.0, 1175.0, 287.0], [2422.0, 2234.0, 725.0, 94.0],
                  [331.0, -2.0, 0, 42.0], [27455.0, 1524.0, 14956.0, 0], [505.9, -272.76, 318.7, 0],
                  [-1051.0, -2745.0, 2812.0, 0]]

cf_url = r'https://www.sec.gov/Archives/edgar/data/1692819/000169281918000018/R5.htm'

# CF Test
#print(sp.cf_htm(cf_url, headers, get_sum_per(cf_url)))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

div_url_list_htm = [r'https://www.sec.gov/Archives/edgar/data/1403161/000140316120000070/R7.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000140316119000050/R7.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000140316118000055/R7.htm',
                    r'https://www.sec.gov/Archives/edgar/data/1403161/000140316115000013/R7.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000138410812000011/R7.htm', r'https://www.sec.gov/Archives/edgar/data/1403161/000140316114000017/R7.htm',
                    r'https://www.sec.gov/Archives/edgar/data/1403161/000140316116000058/R7.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000156459021039151/R99.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000156459019027952/R107.htm',
                    r'https://www.sec.gov/Archives/edgar/data/789019/000156459017014900/R110.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000119312515272806/R112.htm', r'https://www.sec.gov/Archives/edgar/data/789019/000119312513310206/R102.htm',
                    r'https://www.sec.gov/Archives/edgar/data/789019/000119312512316848/R104.htm', r'https://www.sec.gov/Archives/edgar/data/320193/000032019320000096/R6.htm', r'https://www.sec.gov/Archives/edgar/data/320193/000032019318000145/R8.htm',
                    r'https://www.sec.gov/Archives/edgar/data/354950/000035495021000089/R13.htm', r'https://www.sec.gov/Archives/edgar/data/354950/000035495017000005/R7.htm', r'https://www.sec.gov/Archives/edgar/data/354950/000035495013000008/R7.htm',
                    r'https://www.sec.gov/Archives/edgar/data/909832/000090983220000017/R43.htm', r'https://www.sec.gov/Archives/edgar/data/1538990/000155837018001020/R18.htm', r'https://www.sec.gov/Archives/edgar/data/1538990/000155837017001015/R19.htm',
                    r'https://www.sec.gov/Archives/edgar/data/319201/000031920121000029/R7.htm', r'https://www.sec.gov/Archives/edgar/data/319201/000031920119000031/R7.htm', r'https://www.sec.gov/Archives/edgar/data/1053507/000105350721000026/R50.htm',
                    r'https://www.sec.gov/Archives/edgar/data/1053507/000105350716000018/R47.htm', r'https://www.sec.gov/Archives/edgar/data/1053507/000119312515059026/R51.htm', r'https://www.sec.gov/Archives/edgar/data/1287865/000156459021009901/R7.htm',
                    r'https://www.sec.gov/Archives/edgar/data/1553023/000155302315000039/R19.htm', r'https://www.sec.gov/Archives/edgar/data/1677576/000155837021001890/R11.htm', r'https://www.sec.gov/Archives/edgar/data/1253986/000110465921025551/R25.htm',
                    r'https://www.sec.gov/Archives/edgar/data/1253986/000104746914000903/R25.htm', r'https://www.sec.gov/Archives/edgar/data/1253986/000104746913001165/R23.htm', r'https://www.sec.gov/Archives/edgar/data/1253986/000104746912002036/R23.htm',
                    r'https://www.sec.gov/Archives/edgar/data/753308/000075330821000014/R8.htm', r'https://www.sec.gov/Archives/edgar/data/773840/000077384021000015/R6.htm', r'https://www.sec.gov/Archives/edgar/data/773840/000093041319000366/R113.htm',
                    r'https://www.sec.gov/Archives/edgar/data/1501585/000150158513000003/R78.htm', r'https://www.sec.gov/Archives/edgar/data/732717/000073271721000012/R10.htm', r'https://www.sec.gov/Archives/edgar/data/732717/000156276220000064/R9.htm',
                    r'https://www.sec.gov/Archives/edgar/data/1166691/000119312514047522/R30.htm', r'https://www.sec.gov/Archives/edgar/data/1800/000110465921025751/R9.htm', r'https://www.sec.gov/Archives/edgar/data/6281/000000628119000144/R97.htm',
                    r'https://www.sec.gov/Archives/edgar/data/4904/000000490421000010/R4.htm', r'https://www.sec.gov/Archives/edgar/data/91142/000009114221000025/R104.htm', r'https://www.sec.gov/Archives/edgar/data/91142/000119312515050825/R103.htm',
                    r'https://www.sec.gov/Archives/edgar/data/91142/000119312512081738/R106.htm', r'https://www.sec.gov/Archives/edgar/data/1443646/000144364619000093/R104.htm', r'https://www.sec.gov/Archives/edgar/data/1443646/000144364614000015/R87.htm',
                    r'https://www.sec.gov/Archives/edgar/data/1443646/000144364615000018/R87.htm', r'https://www.sec.gov/Archives/edgar/data/1443646/000144364617000050/R89.htm', r'https://www.sec.gov/Archives/edgar/data/1490892/000155837021000757/R56.htm',
                    r'https://www.sec.gov/Archives/edgar/data/1490892/000155837017000503/R55.htm', r'https://www.sec.gov/Archives/edgar/data/1490892/000104746914000899/R51.htm', r'https://www.sec.gov/Archives/edgar/data/1297996/000155837021002191/R99.htm',
                    r'https://www.sec.gov/Archives/edgar/data/1297996/000129799618000026/R84.htm', r'https://www.sec.gov/Archives/edgar/data/1297996/000129799615000010/R87.htm', r'https://www.sec.gov/Archives/edgar/data/1297996/000119312514078397/R95.htm',
                    r'https://www.sec.gov/Archives/edgar/data/921825/000092182521000006/R64.htm', r'https://www.sec.gov/Archives/edgar/data/34903/000003490319000013/R64.htm', r'https://www.sec.gov/Archives/edgar/data/318154/000031815418000004/R92.htm',
                    r'https://www.sec.gov/Archives/edgar/data/318154/000031815415000005/R96.htm', r'https://www.sec.gov/Archives/edgar/data/886982/000119312518056383/R156.htm', r'https://www.sec.gov/Archives/edgar/data/886982/000119312514073792/R106.htm',
                    r'https://www.sec.gov/Archives/edgar/data/886982/000119312512085822/R26.htm', r'https://www.sec.gov/Archives/edgar/data/55785/000005578512000021/R119.htm', r'https://www.sec.gov/Archives/edgar/data/1336920/000133692017000014/R8.htm',
                    r'https://www.sec.gov/Archives/edgar/data/1336920/000133692016000076/R7.htm', r'https://www.sec.gov/Archives/edgar/data/59478/000005947818000089/R92.htm', r'https://www.sec.gov/Archives/edgar/data/59478/000005947813000007/R35.htm',
                    r'https://www.sec.gov/Archives/edgar/data/1059556/000119312518058986/R117.htm', r'https://www.sec.gov/Archives/edgar/data/1059556/000119312512081639/R111.htm', r'https://www.sec.gov/Archives/edgar/data/1164727/000155837020001041/R11.htm',
                    r'https://www.sec.gov/Archives/edgar/data/1705696/000170569619000064/R77.htm', r'https://www.sec.gov/Archives/edgar/data/1705696/000170569620000048/R95.htm', r'https://www.sec.gov/Archives/edgar/data/888491/000088849120000007/R115.htm',
                    r'https://www.sec.gov/Archives/edgar/data/888491/000088849119000006/R116.htm', r'https://www.sec.gov/Archives/edgar/data/78003/000007800320000014/R42.htm', r'https://www.sec.gov/Archives/edgar/data/1413329/000141332913000019/R49.htm',
                    r'https://www.sec.gov/Archives/edgar/data/1034054/000103405421000003/R87.htm', r'https://www.sec.gov/Archives/edgar/data/829224/000082922421000086/R5.htm', r'https://www.sec.gov/Archives/edgar/data/1063761/000155837020001135/R47.htm',
                    r'https://www.sec.gov/Archives/edgar/data/1479094/000104746914001398/R51.htm', r'https://www.sec.gov/Archives/edgar/data/1692819/000169281921000004/R105.htm']

div_answers_htm = [1.2, 1.0, 0.825,
                   0.48, 0.88, 1.6,
                   0.56, 2.24, 1.84,
                   1.56, 1.24, 0.92,
                   0.8, 0.795, 2.72,
                   6.0, 2.76, 1.16,
                   2.7, 1.2, 1.12,
                   3.6, 3.0, 4.53,
                   1.81, 1.4, 1.08,
                   0.84, 4.47, 1.23,
                   0.5, 0.285, '---',
                   1.4, 3.63, 3.055,
                   0.1, 2.08, 2.05,
                   0.78, 1.53, 2.1,
                   2.84, 0.98, 0.6,
                   0.6, 0.8, 2.4,
                   1.46, '---', 4.88,
                   2.36, 1.16, 4.48,
                   3.72, 3.32, 3.12,
                   1.0, 4.02, 4.6,
                   2.62, 2.9, 2.05,
                   '---', 2.8, 14.92,
                   1.28, 2.08, 1.96,
                   1.52, '---', 1.44,
                   0.997, 1.17, '---',
                   2.64, 1.46, 3.24,
                   1.86, 2.29, 8.3,
                   1.2, 0.54]

div_url = r'https://www.sec.gov/Archives/edgar/data/1618921/000114036116083198/R83.htm'

def get_sum_per_div(url):
    obj = re.search(r'(/R\d?\d?\d\.htm)', str(url))
    result = obj.group(1)
    length = len(result) * -1
    sum_url = url[:length] + '/R1.htm'
    fy, per, name = sp.sum_htm(sum_url, headers)
    return per

# Div Test
#print(sp.div_htm(div_url, headers, get_sum_per_div(div_url)))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

catch_url_list_htm = [r'https://www.sec.gov/Archives/edgar/data/100493/000010049320000132/R19.htm', r'https://www.sec.gov/Archives/edgar/data/100493/000010049312000065/R17.htm', r'https://www.sec.gov/Archives/edgar/data/100493/000119312511317791/R17.htm',
                      r'https://www.sec.gov/Archives/edgar/data/318154/000031815418000004/R22.htm', r'https://www.sec.gov/Archives/edgar/data/318154/000031815415000005/R23.htm', r'https://www.sec.gov/Archives/edgar/data/318154/000119312512086670/R22.htm',
                      r'https://www.sec.gov/Archives/edgar/data/1166691/000119312514047522/R55.htm', r'https://www.sec.gov/Archives/edgar/data/1063761/000155837021001700/R45.htm']

catch_answers_htm = [[5.86, 1.725, '---'], [1.58, 0.16, '---'], [1.97, 0.16, '---'],
                     ['---', 4.6, '---'], ['---', 2.44, '---'], ['---', 1.12, '---'],
                     [2.56, '---', 2665000], ['---', 6.0, 308737.62]]

catch_url = r'https://www.sec.gov/Archives/edgar/data/914208/000091420821000331/R105.htm'

# Catch Test
#print(sp.eps_catch_htm(catch_url, headers, get_sum_per_div(catch_url)))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

share_catch_url_list_htm = [r'https://www.sec.gov/Archives/edgar/data/874761/000087476121000015/R7.htm', r'https://www.sec.gov/Archives/edgar/data/764180/000076418020000018/R3.htm', r'https://www.sec.gov/Archives/edgar/data/1166691/000116669121000008/R7.htm',
                            r'https://www.sec.gov/Archives/edgar/data/825542/000154638020000045/R8.htm', r'https://www.sec.gov/Archives/edgar/data/40533/000004053321000010/R63.htm', r'https://www.sec.gov/Archives/edgar/data/55785/000005578513000011/R5.htm',
                            r'https://www.sec.gov/Archives/edgar/data/1137774/000119312514073074/R134.htm', r'https://www.sec.gov/Archives/edgar/data/1137774/000113777421000049/R160.htm', r'https://www.sec.gov/Archives/edgar/data/1137774/000119312513071582/R140.htm',
                            r'https://www.sec.gov/Archives/edgar/data/1063761/000155837021001700/R2.htm']

share_catch_answers_htm = [665370.13, 1857981.56, 4580656.18,
                           55800.0, 287913.0, 428000.0,
                           461100.0, 396400.0, 463000.0,
                           '---']

catch_url = r'https://www.sec.gov/Archives/edgar/data/1618921/000114036115038791/R2.htm'

# Catch Test
#print(sp.share_catch_htm(catch_url, headers, get_sum_per_div(catch_url)))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

sum_url = r'https://www.sec.gov/Archives/edgar/data/726728/000072672811000013/R28.xml'

# XML sum test
#print(sp.sum_xml(sum_url, headers))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# rev, gross, research, oi, net, eps, share_sum, div, ffo
rev_url_list_xml = [r'https://www.sec.gov/Archives/edgar/data/1403161/000119312510265236/R4.xml', r'https://www.sec.gov/Archives/edgar/data/320193/000119312510238044/R2.xml', r'https://www.sec.gov/Archives/edgar/data/909832/000119312510230379/R4.xml',
                    r'https://www.sec.gov/Archives/edgar/data/77476/000119312511040427/R2.xml', r'https://www.sec.gov/Archives/edgar/data/764180/000119312511045778/R2.xml', r'https://www.sec.gov/Archives/edgar/data/200406/000095012311018128/R4.xml',
                    r'https://www.sec.gov/Archives/edgar/data/318154/000095012311018800/R3.xml', r'https://www.sec.gov/Archives/edgar/data/1053507/000119312511050129/R4.xml', r'https://www.sec.gov/Archives/edgar/data/1166691/000119312511047243/R4.xml',
                    r'https://www.sec.gov/Archives/edgar/data/874761/000119312511047720/R2.xml', r'https://www.sec.gov/Archives/edgar/data/315189/000110465910063219/R1.xml', r'https://www.sec.gov/Archives/edgar/data/1289490/000104746911001398/R3.xml',
                    r'https://www.sec.gov/Archives/edgar/data/886982/000095012311020067/R2.xml', r'https://www.sec.gov/Archives/edgar/data/48465/000110465910063729/R3.xml', r'https://www.sec.gov/Archives/edgar/data/1020569/000104746911001600/R3.xml',
                    r'https://www.sec.gov/Archives/edgar/data/1059556/000119312511047974/R2.xml', r'https://www.sec.gov/Archives/edgar/data/751364/000119312511045441/R4.xml', r'https://www.sec.gov/Archives/edgar/data/726728/000072672811000013/R3.xml',
                    r'https://www.sec.gov/Archives/edgar/data/1137774/000119312511047467/R4.xml']

rev_answers_xml = [[8065.0, 7993.0, 0, 4589.0, 2966.0, 4.01, 1096000, '---', 3231.0], [65225.0, 25684.0, 1782.0, 18385.0, 14013.0, 15.15, 924712, '---', 14013.0], [77946.0, 9951.0, 0, 2077.0, 1303.0, 2.92, 445970, 0.77, 1303.0],
                   [57838.0, 31263.0, 0, 8332.0, 6320.0, 3.91, '---', 1.89, 6320.0], [24363.0, 9188.0, 0, 6228.0, 3905.0, 1.87, '---', '---', 3905.0], [61587.0, 42795.0, 6844.0, 16947.0, 13334.0, 4.78, 2788800, 2.11, 13334.0],
                   [15053.0, 12833.0, 2894.0, 5545.0, 4627.0, 4.79, 965000, '---', 4627.0], [1985.34, 1537.71, 0, 784.38, 372.94, 0.92, 404072.0, '---', 833.67], [37937.0, 35781.0, 0, 7980.0, 3635.0, 1.29, '---', 0.378, 3635.0],
                   [16647.0, 3964.0, 0, 1044.0, 9.0, 0.01, '---', '---', 1251.0], [26004.6, 8605.8, 1052.4, 3025.2, 1865.0, 4.35, 428600, 1.16, 1865.0], [281.5, 195.33, 0, 90.81, 26.33, 0.3, 92050.0, 0.4, 76.68],
                   [39161.0, 36880.0, 0, 12892.0, 8354.0, 13.18, 585300.0, 1.4, 8354.0], [7220.72, 1238.74, 0, 646.58, 395.59, 2.92, 135349.0, '---', 395.59], [3127.55, 1873.35, 0, 323.55, -53.9, -0.27, 201991.0, '---', 580.31],
                   [2032.0, 1427.2, 0, 772.8, 507.8, 2.15, 236600.0, '---', 574.1], [229.06, 163.88, 0, 134.91, 66.21, 0.8, 82849.0, '---', 125.36], [345.01, 251.77, 0, 121.42, 106.53, 1.01, 105943.0, '---', 202.85],
                   [38414.0, 13731.0, 0, 4422.0, 3195.0, 5.75, '---', 1.15, 3195.0]]

rev_url = r'https://www.sec.gov/Archives/edgar/data/1090727/000119312511049356/R4.xml'

# XML rev test
#print(sp.rev_xml(rev_url, headers))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# cash, cur_assets, assets, debt, cur_liabilities, liabilities, equity
bs_url_list_xml = [r'https://www.sec.gov/Archives/edgar/data/1403161/000119312510265236/R2.xml', r'https://www.sec.gov/Archives/edgar/data/1018724/000119312511016253/R5.xml', r'https://www.sec.gov/Archives/edgar/data/320193/000119312510238044/R3.xml',
                   r'https://www.sec.gov/Archives/edgar/data/354950/000119312511076501/R3.xml', r'https://www.sec.gov/Archives/edgar/data/909832/000119312510230379/R2.xml', r'https://www.sec.gov/Archives/edgar/data/886982/000095012311020067/R3.xml',
                   r'https://www.sec.gov/Archives/edgar/data/732712/000119312511049476/R3.xml', r'https://www.sec.gov/Archives/edgar/data/1099590/000095012311018724/R2.xml', r'https://www.sec.gov/Archives/edgar/data/1410636/000119312511047938/R2.xml',
                   r'https://www.sec.gov/Archives/edgar/data/18230/000110465911008938/R2.xml', r'https://www.sec.gov/Archives/edgar/data/751364/000119312511045441/R2.xml', r'https://www.sec.gov/Archives/edgar/data/1137774/000119312511047467/R2.xml',
                   r'https://www.sec.gov/Archives/edgar/data/1063761/000104746911001321/R1.xml', r'https://www.sec.gov/Archives/edgar/data/86312/000104746911001001/R2.xml']

bs_answers_xml = [[3867.0, 8734.0, 10483.0, 32.0, 3498.0, 8394.0, 25011.0], [3777.0, 13747.0, 17448.0, 0, 10372.0, 11933.0, 6864.0], [11261.0, 41678.0, 74100.0, 0, 20722.0, 27392.0, 47791.0],
                  [545.0, 13479.0, 38938.0, 8707.0, 10122.0, 21236.0, 18889.0], [3214.0, 11708.0, 23815.0, 2141.0, 10063.0, 12885.0, 10829.0], [39788.0, 526320.0, 911332.0, 174399.0, 659577.0, 833976.0, 77356.0],
                  [6668.0, 22348.0, 198017.0, 45252.0, 30597.0, 133093.0, 86912.0], [56.83, 101.64, 205.04, 0.29, 88.39, 97.96, 171.72], [13.11, 534.31, 12829.08, 5410.27, 774.51, 9947.5, 4132.27],
                  [3592.0, 31810.0, 60601.0, 20437.0, 22020.0, 52695.0, 10864.0], [2.05, 77.24, 2713.58, 1133.68, 7.34, 1184.8, 1527.48], [12915.0, 12915.0, 539854.0, 23653.0, 248371.0, 506926.0, 32415.0],
                  [796.72, 1874.46, 24857.43, 17473.76, 993.74, 19138.21, 4830.78], [200.0, 23372.0, 101816.0, 6611.0, 67802.0, 79706.0, 25475.0]]

bs_url = r'https://www.sec.gov/Archives/edgar/data/86312/000104746911001001/R2.xml'

# XML bs test
#print(sp.bs_xml(bs_url, headers))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# fcf, buyback, divpaid, sbc
cf_url_list_xml = [r'https://www.sec.gov/Archives/edgar/data/1403161/000119312510265236/R8.xml', r'https://www.sec.gov/Archives/edgar/data/320193/000119312510238044/R6.xml', r'https://www.sec.gov/Archives/edgar/data/909832/000119312510230379/R7.xml',
                   r'https://www.sec.gov/Archives/edgar/data/97476/000114036111012312/R7.xml', r'https://www.sec.gov/Archives/edgar/data/66740/000110465911007845/R8.xml', r'https://www.sec.gov/Archives/edgar/data/773840/000093041311000961/R5.xml',
                   r'https://www.sec.gov/Archives/edgar/data/1166691/000119312511047243/R5.xml', r'https://www.sec.gov/Archives/edgar/data/18230/000110465911008938/R6.xml', r'https://www.sec.gov/Archives/edgar/data/78003/000119312511048877/R6.xml']

cf_answers_xml = [[2450.0, 944.0, 368.0, 131.0], [16590.0, -912.0, 0, 879.0], [1725.0, 358.0, 338.0, 190.0],
                  [2621.0, 2047.0, 592.0, 190.0], [4083.0, 854.0, 1500.0, 274.0], [3552.0, -195.0, 944.0, 164.0],
                  [6218.0, 1166.0, 1064.0, 300.0], [3434.0, -296.0, 1084.0, 0], [9941.0, 1000.0, 6088.0, 405.0]]

cf_url = r'https://www.sec.gov/Archives/edgar/data/78003/000119312511048877/R6.xml'

# XML cf Test
#print(sp.cf_xml(cf_url, headers))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

div_url_list_xml = [r'https://www.sec.gov/Archives/edgar/data/789019/000119312510171791/R101.xml', r'https://www.sec.gov/Archives/edgar/data/1403161/000119312510265236/R6.xml', r'https://www.sec.gov/Archives/edgar/data/354950/000119312511076501/R6.xml',
                    r'https://www.sec.gov/Archives/edgar/data/2969/000119312510266784/R158.xml', r'https://www.sec.gov/Archives/edgar/data/825542/000095012310108803/R8.xml', r'https://www.sec.gov/Archives/edgar/data/1045450/000119312511051389/R30.xml',
                    r'https://www.sec.gov/Archives/edgar/data/40533/000119312511039769/R68.xml', r'https://www.sec.gov/Archives/edgar/data/55785/000119312511043300/R122.xml', r'https://www.sec.gov/Archives/edgar/data/895421/000119312511050049/R172.xml',
                    r'https://www.sec.gov/Archives/edgar/data/316709/000119312511046932/R116.xml']

div_answers_xml = [0.52, 0.5, 0.945,
                   1.92, 0.625, '---',
                   1.68, 2.64, 0.2,
                   0.24]

div_url = r'https://www.sec.gov/Archives/edgar/data/1090727/000119312511049356/R75.xml'

# XML Div test
#print(sp.div_xml(div_url, headers))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

catch_url_list_xml = [r'https://www.sec.gov/Archives/edgar/data/100493/000119312510265708/R26.xml', r'https://www.sec.gov/Archives/edgar/data/40533/000119312511039769/R53.xml',
                      r'https://www.sec.gov/Archives/edgar/data/1063761/000104746911001321/R46.xml']

catch_answers_xml = [[2.06, 0.16, '---'], ['---', '---', 385236.0],
                     ['---', 2.6, 291350.47]]

catch_url = r'https://www.sec.gov/Archives/edgar/data/1063761/000104746911001321/R46.xml'

# Catch Test
#print(sp.eps_catch_xml(catch_url, headers))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

share_catch_url_list_xml = [r'https://www.sec.gov/Archives/edgar/data/874761/000119312511047720/R5.xml', r'https://www.sec.gov/Archives/edgar/data/764180/000119312511045778/R4.xml', r'https://www.sec.gov/Archives/edgar/data/1166691/000119312511047243/R3.xml',
                            r'https://www.sec.gov/Archives/edgar/data/1137774/000119312511047467/R134.xml', r'https://www.sec.gov/Archives/edgar/data/1090727/000119312511049356/R3.xml']

share_catch_answers_xml = [787607.24, 2088739.67, 2776499.17,
                           483800.0, 991000.0]

catch_url = r'https://www.sec.gov/Archives/edgar/data/1090727/000119312511049356/R3.xml'

def get_sum_per_div_xml(url):
    obj = re.search(r'(/R\d?\d?\d\.xml)', str(url))
    result = obj.group(1)
    length = len(result) * -1
    sum_url = url[:length] + '/R1.xml'
    fy, per, name = sp.sum_xml(sum_url, headers)
    return per

# Catch Test
#print(sp.share_catch_xml(catch_url, headers, get_sum_per_div_xml(catch_url)))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

tickers = ['AAPL', 'CRM', 'TTD', 'GILD', 'MKC']

per = [['Sep. 26, 2020', 'Sep. 28, 2019', 'Sep. 29, 2018', 'Sep. 30, 2017', 'Sep. 24, 2016', 'Sep. 26, 2015', 'Sep. 27, 2014', 'Sep. 28, 2013', 'Sep. 29, 2012', 'Sep. 24, 2011', 'Sep. 25, 2010'],
       ['Jan. 31, 2021', 'Jan. 31, 2020', 'Jan. 31, 2019', 'Jan. 31, 2018', 'Jan. 31, 2017', 'Jan. 31, 2016', 'Jan. 31, 2015', 'Jan. 31, 2014', 'Jan. 31, 2013', 'Jan. 31, 2012', 'Jan. 31, 2011'],
       ['Dec. 31, 2020', 'Dec. 31, 2019', 'Dec. 31, 2018', 'Dec. 31, 2017', 'Dec. 31, 2016'],
       ['Dec. 31, 2020', 'Dec. 31, 2019', 'Dec. 31, 2018', 'Dec. 31, 2017', 'Dec. 31, 2016', 'Dec. 31, 2015', 'Dec. 31, 2014', 'Dec. 31, 2013', 'Dec. 31, 2012', 'Dec. 31, 2011', 'Dec. 31, 2010'],
       ['Nov. 30, 2020', 'Nov. 30, 2019', 'Nov. 30, 2018', 'Nov. 30, 2017', 'Nov. 30, 2016', 'Nov. 30, 2015', 'Nov. 30, 2014', 'Nov. 30, 2013', 'Nov. 30, 2012', 'Nov. 30, 2011', 'Nov. 30, 2010']]

index = 4    # Change me!

answers = [[1, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 28.0, 28.0, 28.0, 28.0],
           [1, 1, 1, 1, 1, 1, 1, 1, 4.0, 4.0, 4.0],
           [10.0, 10.0, 10.0, 10.0, 10.0],
           [1, 1, 1, 1, 1, 1, 1, 1, 2.0, 2.0, 2.0],
           [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]]

def get_splits(ticker):
    stock = yf.Ticker(ticker)
    splits = stock.splits
    return splits

# Split Factor Test
#print(eq.split_factor_calc(get_splits(tickers[index]), per[index]))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

tickers = ['COR', 'CLX', 'EPR']

divpaid = [[236.3, 223.9, 193.04, 172.48, 108.81, 87.42, 73.77, 57.1, 33.1, 23.79],
           [558.0, 533.0, 490.0, 450.0, 412.0, 398.0, 385.0, 368.0, 335.0, 315.0, 303.0],
           [172.46, 367.32, 342.31, 311.72, 265.66, 233.07, 207.64, 197.92, 162.78, 157.84, 146.32]]

divs = [[4.88, 4.64, 4.02, 3.04, 2.36, 1.7, 1.47, 1.16, 0.81, '---'],
        [4.49, 4.29, 3.84, 3.6, 3.24, 3.11, 2.99, 2.87, 2.63, '---', '---'],
        [1.515, 4.5, 4.32, 4.08, 3.84, 3.63, 3.42, '---', 3.0, 2.8, '---']]

pers = [['Dec. 31, 2020', 'Dec. 31, 2019', 'Dec. 31, 2018', 'Dec. 31, 2017', 'Dec. 31, 2016', 'Dec. 31, 2015', 'Dec. 31, 2014', 'Dec. 31, 2013', 'Dec. 31, 2012', 'Dec. 31, 2011'],
        ['Jun. 30, 2021', 'Jun. 30, 2020', 'Jun. 30, 2019', 'Jun. 30, 2018', 'Jun. 30, 2017', 'Jun. 30, 2016', 'Jun. 30, 2015', 'Jun. 30, 2014', 'Jun. 30, 2013', 'Jun. 30, 2012', 'Jun. 30, 2011'],
        ['Dec. 31, 2020', 'Dec. 31, 2019', 'Dec. 31, 2018', 'Dec. 31, 2017', 'Dec. 31, 2016', 'Dec. 31, 2015', 'Dec. 31, 2014', 'Dec. 31, 2013', 'Dec. 31, 2012', 'Dec. 31, 2011', 'Dec. 31, 2010']]

div_catch_answers = [[4.88, 4.64, 4.02, 3.04, 2.36, 1.7, 1.47, 1.16, 0.81, 0.57],
                     [4.49, 4.29, 3.84, 3.6, 3.24, 3.11, 2.99, 2.87, 2.63, 2.44, 2.25],
                     [1.515, 4.5, 4.32, 4.08, 3.84, 3.63, 3.42, 3.157, 3.0, 2.8, 2.6]]

index = 0       # Change me!

def yf_div_catch_test(ticker, divpaid, divs, pers):
    for index, (dividend, div_paid) in enumerate(zip(divs, divpaid)):
        if div_paid > 0 and dividend == '---':
            if index == 0:
                divs[index] = eq.yf_div_catch(ticker, pers[index])
            else:
                divs[index] = eq.yf_div_catch(ticker, pers[index], pers[index - 1], divs[index - 1])
    
    return divs

# YF Div Catch Test
#print(yf_div_catch_test(tickers[index], divpaid[index], divs[index], pers[index]))


'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# Div finder
base_url = 'https://www.sec.gov/Archives/edgar/data/1692819/000169281921000004/'
div = 1.08
def div_finder_htm(base_url, div):
    for i in range(150):
        url = base_url + 'R' + str(i) + '.htm'
        content = requests.get(url, headers=headers).content
        soup = BeautifulSoup(content, 'html.parser')
        if 'Dividend' in str(soup) or str(div) in str(soup):
            print(url)

#div_finder_htm(base_url, div)

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# Div finder
base_url = 'https://www.sec.gov/Archives/edgar/data/1710583/000171058318000034/'
div = 0.014
def div_finder_xml(base_url, div):
    for i in range(150):
        url = base_url + 'R' + str(i) + '.xml'
        content = requests.get(url, headers=headers).content
        soup = BeautifulSoup(content, 'xml')
        if 'Dividend' in str(soup) or str(div) in str(soup):
            print(url)

#div_finder_xml(base_url, div)

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# Report name finder

def pull_filing_2(xml_summary):
    content = requests.get(xml_summary, headers=headers).content
    soup = BeautifulSoup(content, 'lxml')

    # Find the all of the individual reports submitted
    reports = soup.find('myreports')
    assert reports != None
    return reports


def name_finder(xml_summary):
    reports = pull_filing_2(xml_summary)
    for report in reports.find_all('report')[:-1]:
        print(report.shortname.text.upper())


xml_summary = r'https://www.sec.gov/Archives/edgar/data/1479094/000104746914001398/FilingSummary.xml'

#print(name_finder(xml_summary))