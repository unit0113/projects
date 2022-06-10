import plotly.express as px
import pandas as pd
from datetime import date


today = date.today().strftime('%d %B %Y')

cols = [0, 2, 4, 8, 11 , 12, 13, 14, 21, 23]
df = pd.read_excel(r"C:\Users\unit0\Desktop\Portfolio Tracker.xlsm", sheet_name='Analysis', header=1, usecols=cols)
df = df.drop(df.index[51:57])
df = df.truncate(after=105)

# Make percentages actually percentages
df['Daily Change (%)'] = df['Daily Change (%)'].apply(lambda x: x*100)
df['Gain (%)'] = df['Gain (%)'].apply(lambda x: x*100)
df['Div Yield'] = df['Div Yield'].apply(lambda x: x*100)

# Format strings
df['Daily Change (%) Numerical'] = df['Daily Change (%)']
df['Daily Change (%)'] = df['Daily Change (%)'].apply(lambda x: f'{x:.2f}%')
df['Gain (%)'] = df['Gain (%)'].apply(lambda x: f'{x:.2f}%')
df['Div Yield'] = df['Div Yield'].apply(lambda x: f'{x:.2f}%')
df['Market Value Numerical'] = df['Market Value']
df['Market Value'] = df['Market Value'].apply(lambda x: f'${x:.2f}')
df['Cost Basis'] = df['Cost Basis'].apply(lambda x: f'${x:.2f}')
df['Gain ($)'] = df['Gain ($)'].apply(lambda x: f'${x:.2f}')
df['Yearly Div'] = df['Yearly Div'].apply(lambda x: f'${x:.2f}')

# Create figure
fig = px.treemap(df, path=[px.Constant("Stock Portfolio"), 'Account', 'Sector', 'Ticker'],
                 values='Market Value Numerical',
                 color='Daily Change (%) Numerical',
                 color_continuous_scale=["red", "black", "green"],
                 color_continuous_midpoint=0,
                 hover_data={'Sector': True, 'Market Value': True, 'Gain (%)': True, 'Div Yield': True, 'Yearly Div': True},
                 hover_name='Ticker',
                 branchvalues='total',
                 custom_data=['Sector', 'Market Value', 'Gain (%)', 'Div Yield', 'Yearly Div', 'Ticker', 'Daily Change (%)']
                 )

# Edit text and hover text templates
fig.data[0].hovertemplate = "<br>".join([
        "%{customdata[5]}",
        "%{customdata[6]}",
        "Sector: %{customdata[0]}",
        "Market Value: %{customdata[1]}",
        "Gain: %{customdata[2]}",
        "Divident Yield: %{customdata[3]}",
        "Yearly Dividend: %{customdata[4]}",
    ])

fig.data[0].texttemplate ="<br>".join([
        "%{customdata[5]}",
        "%{customdata[6]}"
    ])

fig.update_traces(textposition='middle center')
fig.update(layout_coloraxis_showscale=False)
fig.update_layout(margin = dict(t=0, l=0, r=0, b=0),
        plot_bgcolor='#3a3b3c',
        paper_bgcolor='#3a3b3c',
        title_text = f'Portfolio Status {today}',
        uniformtext_minsize=18, uniformtext_mode='hide' # check for other modes
    )
fig.show()