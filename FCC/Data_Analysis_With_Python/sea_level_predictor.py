import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def draw_plot():
    # Read data from file
    df = pd.read_csv('epa-sea-level.csv')

    # Create scatter plot
    plt.scatter(df['Year'], df['CSIRO Adjusted Sea Level'], label='Observed Data')

    # Create first line of best fit
    res = linregress(df['Year'], df['CSIRO Adjusted Sea Level'])
    years_extended = np.arange(df['Year'].min(), 2051, 1)
    plt.plot(years_extended, res.intercept + res.slope*years_extended, 'r', label='Total Trend')

    # Create second line of best fit
    line_two_df = df[df['Year'] >= 2000]
    res = linregress(line_two_df['Year'], line_two_df['CSIRO Adjusted Sea Level'])
    years_extended = np.arange(line_two_df['Year'].min(), 2051, 1)
    plt.plot(years_extended, res.intercept + res.slope*years_extended, 'r', label='Recent Trend')

    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Sea Level (inches)')
    plt.title('Rise in Sea Level')
    
    # Save plot and return data for testing (DO NOT MODIFY)
    plt.savefig('sea_level_plot.png')
    return plt.gca()