import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_style('whitegrid')

# Pre-optimization
df1 = pd.read_excel(r"UF\COP3503\lab3\lab3data.xlsx", sheet_name="Pre")
#print(df1.head())
#stackDf1 = df1[df1["DS"] == "Stack"]
#queueDf1 = df1[df1["DS"] == "Queue"]

# Post-optimization
df2 = pd.read_excel(r"UF\COP3503\lab3\lab3data.xlsx", sheet_name="Post")
#print(df2.head())
#stackDf2 = df2[df2["DS"] == "Stack"]
#queueDf2 = df2[df2["DS"] == "Queue"]


#facet = sns.FacetGrid(df1, col = "Type")
#facet.map(sns.lineplot, "N", "Duration")

#sns.pairplot(df1)
"""g = sns.FacetGrid(df1[(df1["Scale Factor"] == 100) & (df1["Operation"] != "Dequeue")], col = "DS", hue = "Operation")
g.map(sns.lineplot, "N", "Duration" )
g.add_legend()
g.set(title="Scale Factor: 100")"""
#p = sns.lineplot(data=df1[(df1["Scale Factor"] == 2) & (df1["Operation"] != "Dequeue")], x='N', y='Duration', hue='Operation')
#p.set_title("Scale Factor: 100, Dequeue Excluded")

# N vs Duration
#p = sns.lineplot(data=df2[df2["Scale Factor"] == 2], x='N', y='Duration', hue='Operation')
#p.set_title("N Analysis; Scale Factor: 2; Post-Optimization")

# Scale Factor vs Duration
#p = sns.lineplot(data=df2[df2["N"] == 100_000_000], x='Scale Factor', y='Duration', hue='Operation')
#p.set_title("Scale Factor Analysis; N: 100,000,000; Post-Optimization")

# Scale Factor vs resizes
#p = sns.lineplot(data=df2[df2["N"] == 100_000_000], x='Scale Factor', y='Resizes', hue='Operation')
#p.set_title("Scale Factor Analysis; N: 100,000,000; Post-Optimization")

# N vs resizes
p = sns.lineplot(data=df2[df2["Scale Factor"] == 2], x='N', y='Resizes', hue='Operation')
p.set_title("Memory Analysis; Scale Factor: 2; Post-Optimization")

plt.show(block=True)