import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv(r'DS\Coursera\U_IL\data_viz\ExcelFormattedGISTEMPData2CSV.csv')

sns.set_theme()
sns.set(style="ticks", context="talk", rc={'figure.figsize':(20, 12)})
plt.style.use("dark_background")

ax = sns.lineplot(x='Year', y='value', hue='variable',
                  data=pd.melt(data[['Year', 'Glob', 'NHem', 'SHem']], ['Year']),
                  palette = 'hls')

# Trendline    
z = np.polyfit(data['Year'], data['Glob'], 1)
p = np.poly1d(z)
plt.plot(data['Year'],p(data['Year']),"r--")

plt.xlim(1880)

ax.set_xlabel('Year',fontsize=25)
ax.set_ylabel('Temperature Variation (°C)',fontsize=25)
ax.axes.set_title('Global and Hemispheric Zonal Annual Means', fontsize=35)

ax.legend(title='Region', loc='upper left', labels=['Global', 'Northern Hemisphere', 'Southern Hemisphere'])
plt.setp(ax.get_legend().get_texts(), fontsize='23')
plt.setp(ax.get_legend().get_title(), fontsize='28')

plt.show()

