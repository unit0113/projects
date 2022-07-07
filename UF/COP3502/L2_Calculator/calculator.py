def menu():
    print('Calculator Menu')
    print('---------------')
    print('1. Addition')
    print('2. Subtraction')
    print('3. Multiplication')
    print('4. Division')

    option = int(input('Which operation do you want to perform? '))
    if option < 1 or option > 4:
        print('Error: Invalid selection! Terminating program.')
        quit()

    return option



def main():
    num1 = float(input('Enter first operand: '))
    num2 = float(input('Enter second operand: '))

    option = menu()
    if option == 1:
        result = num1 + num2
    elif option == 2:
        result = num1 - num2
    elif option == 3:
        result = num1 * num2
    elif option == 4:
        result = num1 / num2

    print(f'The result of the operation is {result}. Goodbye!')

if __name__ == "__main__":
    main()