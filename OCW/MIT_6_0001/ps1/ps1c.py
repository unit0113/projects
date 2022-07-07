total_cost = 1_000_000
portion_down_payment = total_cost * 0.25
current_savings = 0
r = 0.04
annual_salary_init = int(input('Enter your starting annual salary: '))
semi_annual_raise = 0.07

steps = 0
months = 36
low, high = 0, 10000
success_flag = True
while current_savings < portion_down_payment - 100 or current_savings > portion_down_payment + 100:
    if high - low <= 1:
        success_flag = False
        break
    steps += 1
    portion_saved_guess = (high + low) // 2
    annual_salary = annual_salary_init
    current_savings = 0
    for month in range(months):
        current_savings += current_savings * (r / 12) + annual_salary * (portion_saved_guess / 10000) / 12

        if month % 6 == 0:
            annual_salary *= (1 + semi_annual_raise)
        
    if current_savings < portion_down_payment:
        low = portion_saved_guess
    else:
        high = portion_saved_guess

if success_flag:
    print(f'Best savings rate: {portion_saved_guess / 10000}')
    print(f'Steps in bisection search: {steps}')
else:
    print('It is not possible to pay the down payment in three years.')