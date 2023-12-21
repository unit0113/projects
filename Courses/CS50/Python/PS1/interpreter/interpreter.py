num1, expression, num2 = input("Enter an expression: ").strip().split()
num1 = float(num1)
num2 = float(num2)

match expression:
    case '+':
        print(num1 + num2)
    case '-':
        print(num1 - num2)
    case '*':
        print(num1 * num2)
    case '/':
        print(num1 / num2)