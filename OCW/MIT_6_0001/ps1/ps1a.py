total_cost = int(input('Enter the cost of your dream home: '))
portion_down_payment = total_cost * 0.25
current_savings = 0
r = 0.04
annual_salary = int(input('Enter your annual salary: '))
portion_saved = float(input('Enter the percent of your salary to save, as a decimal: '))

months = 0
while current_savings < portion_down_payment:
    current_savings += current_savings * (r / 12) + annual_salary * portion_saved / 12
    months += 1

print(f'Number of months: {months}')