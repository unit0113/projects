from cs50 import get_float

# Get amount of change owed from user, value greater than 0
change = -1
while change < 0:
    change = get_float("Change owed: ")

q = 0
d = 0
n = 0
p = 0
    
# Calculate quarters
while change > 0.24:
    q += 1
    change -= 0.25
    change = round(change, 2)
    
# Calculate dimes
while change > 0.09:
    d += 1
    change -= 0.10
    
# Calculate nickles
while change > 0.04:
    n += 1
    change -= 0.05    

# Calculate pennies
while change > 0:
    p += 1
    change -= 0.01  
    
print(q + d + n + p)    