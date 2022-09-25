import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_style('whitegrid')
df1 = pd.read_excel(r"UF\COP3503\lab3\lab3data.xlsx")
#print(df1.head())
#stackDf1 = df1[df1["DS"] == "Stack"]
#queueDf1 = df1[df1["DS"] == "Queue"]

#facet = sns.FacetGrid(df1, col = "Type")
#facet.map(sns.lineplot, "N", "Duration")

#sns.pairplot(df1)
"""g = sns.FacetGrid(df1[(df1["Scale Factor"] == 100) & (df1["Operation"] != "Dequeue")], col = "DS", hue = "Operation")
g.map(sns.lineplot, "N", "Duration" )
g.add_legend()
g.set(title="Scale Factor: 100")"""
p = sns.lineplot(data=df1[(df1["Scale Factor"] == 2) & (df1["Operation"] != "Dequeue")], x='N', y='Duration', hue='Operation')
p.set_title("Scale Factor: 100, Dequeue Excluded")

plt.show(block=True)