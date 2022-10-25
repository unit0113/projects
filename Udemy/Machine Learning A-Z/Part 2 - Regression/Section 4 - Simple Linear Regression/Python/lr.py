import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv(r"Udemy\Machine Learning A-Z\Part 2 - Regression\Section 4 - Simple Linear Regression\Python\Salary_Data.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lr = LinearRegression().fit(X_train, y_train)
y_pred = lr.predict(X_test)

plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, lr.predict(X_train))
plt.title("Salary vs Experience")
plt.xlabel("Years Experience")
plt.ylabel("Salary ($)")
plt.show()