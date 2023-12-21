num1 = 'a'
num2 = 'b'

while not num1.isnumeric() or not num2.isnumeric():
    raw_input = input("Fraction: ")
    if '/' not in raw_input or '.' in raw_input:
        continue

    num1, num2 = raw_input.split('/')

    try:
        num1 = int(num1)
        num2 = int(num2)

    except ValueError:
        continue
    except ZeroDivisionError:
        num2 = 'b'
        continue

    if num1 > num2:
        num1 = 'a'
        continue

    percent = 100 * int(num1) / int(num2)
    break

if percent >= 99:
    print('F')
elif percent <= 1:
    print('E')
else:
    print(f'{round(percent)}%')