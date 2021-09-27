import json
import statement_parser as sp
import edgarquery as eq
import math
import re
import yfinance as yf

# Header data for data pull
with open(r'C:\Users\unit0\OneDrive\Desktop\EDGAR\user_agent.txt') as f:
    data = f.read()
    headers = json.loads(data)


sum_url = r'https://www.sec.gov/Archives/edgar/data/320193/000032019320000096/R1.htm'

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
                    r'https://www.sec.gov/Archives/edgar/data/764478/000076447813000014/R4.htm', r'https://www.sec.gov/Archives/edgar/data/1253986/000110465921025551/R4.htm', r'https://www.sec.gov/Archives/edgar/data/753308/000075330821000014/R2.htm']

rev_answers_htm = [[168088, 115856, 20716, 69916, 61271, 8.05, 7608000, '---', 61271], [274515, 104956, 18752, 66288, 57411, 3.28, 17528214, '---', 57411], [215639, 84263, 10045, 60024, 45687, 8.31, 5500281, 2.18, 45687],
                   [170910, 64304, 4475, 48999, 37037, 39.75, 931662, 11.4, 37037], [89950, 55689, 13037, 22326, 21204, 2.71, 7832000, 1.56, 21204], [77849, 57600, 10411, 26764, 21863, 2.58, 8470000, 0.92, 21863],
                   [386064, 152757, 42740, 22899, 21331, 41.83, 510000, '---', 21331], [177866, 65932, 22620, 4106, 3033, 6.15, 493000, '---', 3033], [74452, 20271, 6565, 745, 274, 0.59, 465000, '---', 274],
                   [182527, 97795, 27573, 41224, 40269, 58.61, 687067, '---', 40269], [136819, 77270, 21419, 26321, 30736, 43.7, 703341, '---', 30736], [74989, 46825, 12282, 19360, 16348, 22.84, 715762, '---', 16348],
                   [21846, 21846, 0, 14081, 10866, 4.89, 2479000, '---', 11633], [18358, 18358, 0, 12144, 6699, 2.8, 2654000, '---', 7255], [12702, 12702, 0, 7697, 5438, 8.62, 902000, '---', 5873],
                   [125843, 82933, 16876, 42959, 39240, 5.06, 7753000, '---', 39240], [156508, 68662, 3381, 55241, 41733, 44.15, 945355, 2.65, 41733], [93580, 60542, 12046, 18161, 12193, 1.48, 8254000, 1.24, 12193],
                   [182795, 70537, 6041, 52503, 39510, 6.45, 6122663, 1.82, 39510], [73723, 56193, 9811, 21763, 16978, 2.0, 8506000, 0.8, 16978], [265595, 101839, 14236, 70898, 59531, 11.91, 5000109, '---', 59531],
                   [110855, 65272, 16625, 26146, 12662, 18.0, 703444, '---', 12662], [83176, 28954, 0, 10469, 6345, 4.71, 1346000, '---', 7996], [7000, 1599, 834, -667, -675, -4.68, 144212, '---', -675],
                   [2013, 456, 232, -61, -74, -0.62, 119421, '---', -74], [166761, 21822, 0, 5435, 4002, 9.02, 443901, '---', 4002], [105156, 13208, 0, 3053, 2039, 4.63, 440512, 8.17, 2039],
                   [99137, 12314, 0, 2759, 1709, 3.89, 439373, 1.03, 1709], [694, 694, 0, 187, 213, 0.84, 252651, '---', 456], [25110, 12711, 0, 10326, -1293, -0.7, 1847143, '---', -1293],
                   [7911, 2703, 0, 702, 532, 4.31, 123471, 1.2, 709], [22859, 18359, 4285, 7537, 5144, 3.13, 1637000, 2.1, 5144], [23362, 19006, 4116, 9674, 7842, 12.88, 609000, '---', 7842],
                   [15582, 13155, 3167, 4312, 3683, 4.04, 912000, '---', 3683], [705, 705, 0, 486, 289, 0.82, 350441, '---', 407], [1249, 1249, 0, 499, 430, 0.81, 530461, '---', 678],
                   [1034, 622, 0, 40, 41, 0.35, 117600, '---', 490], [672, 672, 0, 24, -84, -0.95, 88900, '---', 235], [399, 399, 0, 23, -10, -0.3, 54300, '---', 146],
                   [45085, 10649, 0, -125, -441, -1.3, 338600, '---', -441], [434, 434, 0, 164, 171, 1.41, 133969, '---', 179], [1797, 17997, 0, 5116, 2919, 1.48, 1972297, '---', 2919]]

rev_url = r'https://www.sec.gov/Archives/edgar/data/753308/000075330821000014/R2.htm'

# Rev Test
#print(sp.rev_htm(rev_url, headers, get_sum_per(rev_url)))


'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


# cash, assets, debt, liabilities, equity
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
                   r'https://www.sec.gov/Archives/edgar/data/753308/000075330813000023/R5.htm']

bs_answers_htm = [[16289, 37201, 21071, 44709, 36210], [8162, 26473, 16630, 35219, 34006], [5619, 21735, 15882, 31123, 32912],
                  [1971, 15405, 0, 11156, 27413], [2127, 11656, 0, 8323, 26437], [14224, 276268, 50074, 191791, 141988],
                  [11356, 236780, 66662, 184226, 102330], [7663, 195858, 76073, 168692, 72394], [5595, 154449, 27808, 96140, 80083],
                  [3804, 124693, 12601, 63487, 78944], [6938, 104649, 10713, 54908, 66363], [38016, 323888, 98667, 258549, 65339],
                  [20289, 367304, 97207, 241272, 134047], [13844, 223081, 28987, 120292, 111547], [9815, 111939, 0, 39756, 76615],
                  [26465, 296996, 13932, 97072, 222544], [10715, 177856, 3969, 44793, 152502], [16549, 127745, 3990, 27130, 120331],
                  [2538, 40873, 22349, 38633, 4333], [2133, 48982, 28670, 54352, -3116], [12277, 55556, 7514, 36851, 18284],
                  [6055, 40830, 6487, 27727, 12799], [8185, 54846, 40370, 79366, 13454], [84, 5165, 366, 3722, 1567],
                  [64, 2361, 150, 978, 1393], [1234, 5844, 3470, 6599, 2665], [3805, 37895, 29623, 44029, 22096],
                  [166, 9004, 2213, 3989, 5016], [1746, 26111, 28498, 42453, 4094], [83, 6419, 2909, 3165, 3248],
                  [351, 25818, 9380, 10635, 13976], [340, 7556, 3322, 6178, 1344], [438, 69306, 23969, 51266, 18040],
                  [329, 64439, 23177, 48371, 16068]]

bs_url = r'https://www.sec.gov/Archives/edgar/data/753308/000075330813000023/R5.htm'


# BS Test
#print(sp.bs_htm(bs_url, headers, get_sum_per(rev_url)))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# fcf, debt_pay, buyback, divpaid, sbc
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
                   r'https://www.sec.gov/Archives/edgar/data/1045609/000156459019002872/R7.htm', r'https://www.sec.gov/Archives/edgar/data/97476/000009747614000007/R7.htm']

cf_answers_htm = [[23907, 4461, 4846, 0, 7679], [9704, 537, 7924, 2664, 416], [11995, 2295, 7028, 1918, 327],
                  [5051, 244, 7062, 1350, 221], [6652, 0, 4027, 1006, 172], [4633, 0, 536, 595, 147],
                  [56118, 3750, 25692, 16521, 6118], [38260, 4000, 18401, 13811, 4652], [31378, 7922, 11016, 11845, 3266],
                  [23136, 1500, 13809, 9882, 2574], [24576, 1346, 4429, 7455, 2406], [29321, 0, 3116, 6385, 2244],
                  [73365, 15631, 71478, 14081, 6829], [51147, 5592, 32345, 12769, 4840], [50142, 339, 44270, 11126, 2863],
                  [33269, 0, -831, 0, 1168], [42843, 2100, 31149, 0, 12991], [16109, 13824, 1780, 47, 5203],
                  [24639, 814, 9133, 5180, 2166], [16376, 4113, 465, 6451, 310], [6800, 821, 6748, 2530, 225],
                  [-3, 381, -296, 0, 749],[-2159, 32, -837, 0, 198], [1354, 86, -16, 3560, 285],
                  [432, 160, -691, 353, 5], [246, 96, -477, 163, 7], [8154, 2246, 0, 6290, 0],
                  [1012, 1957, -266, 2477, 235], [-492, 47, -1599, 81, 14], [2937, 6782, 44, 1781, 110],
                  [1804, 4166, 7, 1168, 76], [2972, 1500, 4182, 1175, 287]]

cf_url = r'https://www.sec.gov/Archives/edgar/data/97476/000009747614000007/R7.htm'

# CF Test
#print(sp.cf_htm(cf_url, headers, get_sum_per(rev_url)))

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
                    r'https://www.sec.gov/Archives/edgar/data/1253986/000104746914000903/R25.htm', r'https://www.sec.gov/Archives/edgar/data/1253986/000104746913001165/R23.htm', r'https://www.sec.gov/Archives/edgar/data/1253986/000104746912002036/R23.htm']

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
                   0.5, 0.285, '---']

div_url = r'https://www.sec.gov/Archives/edgar/data/1730168/000173016820000226/R18.htm'

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
                      r'https://www.sec.gov/Archives/edgar/data/318154/000031815418000004/R22.htm', r'https://www.sec.gov/Archives/edgar/data/318154/000031815415000005/R23.htm', r'https://www.sec.gov/Archives/edgar/data/318154/000119312512086670/R22.htm']
eps_list_htm = [5.86, 1.58, 1.97, 2.69, 6.7, 4.04]
index = 5    # Change me!
catch_answers_htm = [[5.86, 1.725], [1.58, 0.16], [1.97, 0.16], [2.69, 4.6], [6.7, 2.44], [4.04, 1.12]]

catch_url = r'https://www.sec.gov/Archives/edgar/data/318154/000119312512086670/R22.htm'

# Catch Test
#print(sp.eps_catch_htm(catch_url_list_htm[index], headers, eps_list_htm[index]))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

sum_url = r'https://www.sec.gov/Archives/edgar/data/1045609/000095012311015736/R1.xml'

# XML sum test
#print(sp.sum_xml(sum_url, headers))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# rev, gross, research, oi, net, eps, share_sum, div, ffo
rev_url_list_xml = [r'https://www.sec.gov/Archives/edgar/data/1403161/000119312510265236/R4.xml', r'https://www.sec.gov/Archives/edgar/data/320193/000119312510238044/R2.xml', r'https://www.sec.gov/Archives/edgar/data/909832/000119312510230379/R4.xml',
                    r'https://www.sec.gov/Archives/edgar/data/77476/000119312511040427/R2.xml', r'https://www.sec.gov/Archives/edgar/data/764180/000119312511045778/R2.xml', r'https://www.sec.gov/Archives/edgar/data/200406/000095012311018128/R4.xml',
                    r'https://www.sec.gov/Archives/edgar/data/318154/000095012311018800/R3.xml', r'https://www.sec.gov/Archives/edgar/data/1053507/000119312511050129/R4.xml']

rev_answers_xml = [[8065, 8065, 0, 4589, 2966, 4.01, 1096000, '---', 2966], [65225, 25684, 1782, 18385, 14013, 15.15, 924712, '---', 14013], [77946, 9951, 0, 2077, 1303, 2.92, 445970, 0.77, 1303],
                   [57838, 31263, 0, 8332, 6320, 3.91, 1616368, 1.89, 6320], [24363, 9188, 0, 6228, 3905, 1.87, 2088235, '---', 3905], [61587, 42795, 6844, '---', 13334, 4.78, 2788800, 2.11, 13334],
                   [15053, 12833, 2894, 5545, 4627, 4.79, 965000, '---', 4627], [1985, 1985, 0, 784, 373, 0.92, 404072, '---', 834]]

rev_url = r'https://www.sec.gov/Archives/edgar/data/1045609/000095012311015736/R4.xml'

# XML rev test
#print(sp.rev_xml(rev_url, headers))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# cash, assets, debt, liabilities, equity
bs_url_list_xml = [r'https://www.sec.gov/Archives/edgar/data/1403161/000119312510265236/R2.xml', r'https://www.sec.gov/Archives/edgar/data/1018724/000119312511016253/R5.xml', r'https://www.sec.gov/Archives/edgar/data/320193/000119312510238044/R3.xml',
                   r'https://www.sec.gov/Archives/edgar/data/354950/000119312511076501/R3.xml', r'https://www.sec.gov/Archives/edgar/data/909832/000119312510230379/R2.xml', r'https://www.sec.gov/Archives/edgar/data/1045609/000095012311015736/R2.xml']

bs_answers_xml = [[3867, 10483, 32, 8394, 25011], [3777, 17448, 0, 11933, 6864], [11261, 74100, 0, 27392, 47791],
                  [545, 38938, 8707, 21236, 18889], [3214, 23815, 2141, 12885, 10829], [198, 7373, 3331, 3671, 3321]]

bs_url = r'https://www.sec.gov/Archives/edgar/data/1045609/000095012311015736/R2.xml'

# XML bs test
#print(sp.bs_xml(bs_url, headers))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# fcf, debt_pay, buyback, divpaid, sbc
cf_url_list_xml = [r'https://www.sec.gov/Archives/edgar/data/1403161/000119312510265236/R8.xml', r'https://www.sec.gov/Archives/edgar/data/320193/000119312510238044/R6.xml', r'https://www.sec.gov/Archives/edgar/data/909832/000119312510230379/R7.xml',
                   r'https://www.sec.gov/Archives/edgar/data/97476/000114036111012312/R7.xml']

cf_answers_xml = [[2450, 16, 944, 368, 131], [16590, 0, -912, 0, 879], [1725, 194, 358, 338, 190],
                  [2621, 0, 2454, 592, 190]]

cf_url = r'https://www.sec.gov/Archives/edgar/data/97476/000114036111012312/R7.xml'

# XML cf Test
#print(sp.cf_xml(cf_url, headers))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

div_url_list_xml = [r'https://www.sec.gov/Archives/edgar/data/789019/000119312510171791/R101.xml', r'https://www.sec.gov/Archives/edgar/data/1403161/000119312510265236/R6.xml', r'https://www.sec.gov/Archives/edgar/data/354950/000119312511076501/R6.xml']

div_answers_xml = [0.52, 0.5, 0.945]

div_url = r'https://www.sec.gov/Archives/edgar/data/354950/000119312511076501/R6.xml'

# XML Div test
#print(sp.div_xml(div_url, headers))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

catch_url_list_xml = [r'https://www.sec.gov/Archives/edgar/data/100493/000119312510265708/R26.xml', r'https://www.sec.gov/Archives/edgar/data/318154/000095012311018800/R23.xml', r'https://www.sec.gov/Archives/edgar/data/318154/000095012311018800/R12.xml']
eps_list_xml = [2.06, 4.79, 4.79]
index = 2    # Change me!
catch_answers_xml = [[2.06, 0.16], [4.79, '---'], [4.79, '---']]

catch_url = r'https://www.sec.gov/Archives/edgar/data/318154/000095012311018800/R12.xml'

# Catch Test
#print(sp.eps_catch_xml(catch_url_list_xml[index], headers, eps_list_xml[index]))

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
tickers = ['AAPL']

per = [['Sep. 26, 2020', 'Sep. 28, 2019', 'Sep. 29, 2018', 'Sep. 30, 2017', 'Sep. 24, 2016', 'Sep. 26, 2015', 'Sep. 27, 2014', 'Sep. 28, 2013', 'Sep. 29, 2012', 'Sep. 24, 2011', 'Sep. 25, 2010']]

index = 0    # Change me!

def get_splits(ticker):
    stock = yf.Ticker(ticker)
    splits = stock.splits
    return splits

# Split Factor Test
#print(eq.split_factor_calc(get_splits(tickers[index]), per[index]))