from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plt
import squarify


url = r'https://companiesmarketcap.com/dow-jones/largest-companies-by-market-cap/'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'lxml')
rows = soup.findChildren('tr')
symbols = []
market_caps = []
numerical_market_caps = []
day_changes = []

for row in rows:
    try:
        symbol = row.find('div', {'class': 'company-code'}).text
        symbols.append(symbol)
        market_cap = row.findAll('td')[2].text
        market_caps.append(market_cap)
        day_change_td = row.findAll('td')[4]
        if 'percentage-green' in str(day_change_td):
            sign = ''
        else:
            sign = '-'
        day_changes.append(sign + day_change_td.text)

        if market_cap.endswith('T'):
            numerical_market_caps.append(float(market_cap[1:-2]) * 10e12)
        elif market_cap.endswith('B'):
            numerical_market_caps.append(float(market_cap[1:-2]) * 10e9)
        elif market_cap.endswith('M'):
            numerical_market_caps.append(float(market_cap[1:-2]) * 10e6)
        
    except:
        pass

print(day_changes)
labels = [f'{symbol}\n({market_cap})\n{day_change}' for symbol, market_cap, day_change in zip(symbols, market_caps, day_changes)]
#colors = [plt.cm.tab20c(i / len(symbols)) for i in range(len(symbols))]
colors = [(0.6451, 0, 0, 1) if change.startswith('-') else (0, 0.6451, 0, 1) for change in day_changes ]
plt.figure(figsize=(24,12))
squarify.plot(sizes=numerical_market_caps, label=labels, color=colors,
              bar_kwargs={'linewidth': 0.5, 'edgecolor': '#111111'})
plt.show()