import math

def add(num1, num2):
    """Dummy function because zybooks can't handle the class method

    Args:
        num1 (float): First operand
        num2 (float): Second operand

    Returns:
        float: result of addition
    """
    
    return num1 + num2


class Calculator:
    def __init__(self):
        self.total_results = 0.0
        self.previous_result = None
        self.num_calculations = 0


    def menu(self):
        """Print menu options. Bridge to other methods if called correctly
        """

        # Print Menu
        print(f'Current Result: {self.previous_result if self.previous_result else 0.0}')
        print()
        print('Calculator Menu')
        print('---------------')
        print('0. Exit Program')
        print('1. Addition')
        print('2. Subtraction')
        print('3. Multiplication')
        print('4. Division')
        print('5. Exponentiation')
        print('6. Logarithm')
        print('7. Display Average')
        print()

        # Get valid input
        option = -1
        while option < 0 or option > 6:
            option = int(input('Enter Menu Selection: '))
            if option < 0 or option > 7:
                print()
                print('Error: Invalid selection!')
                print()

            # Display average data
            elif option == 7:
                if not self.num_calculations:
                    print('Error: No calculations yet to average!')
                else:
                    print(f'Sum of calculations: {self.total_results}')
                    print(f'Number of calculations: {self.num_calculations}')
                    print(f'Average of calculations: {self.total_results / self.num_calculations:.2f}')
                print()

        # Exit
        if option == 0:
            print('Thanks for using this calculator. Goodbye!')
            quit()
        
        # Call appropriate functions
        elif option == 1:
            self.add()

        elif option == 2:
            self.subtract()

        elif option == 3:
            self.multiply()

        elif option == 4:
            self.divide()

        elif option == 5:
            self.exponent()

        elif option == 6:
            self.log()

    
    def get_input(self):
        """Gets operands from user

        Returns:
            tupple: the first and second operands
        """
        num1 = input('Enter first operand: ')
        num2 = input('Enter second operand: ')

        # Grab previous result if requested
        num1 = self.previous_result if num1 == 'RESULT' else float(num1)
        num2 = self.previous_result if num2 == 'RESULT'else float(num2)

        return num1, num2


    def add(self):
        """Add two numbers and update class variables
        """
        num1, num2 = self.get_input()
        self.num_calculations += 1
        self.previous_result = num1 + num2
        self.total_results += self.previous_result

        self.menu()


    def subtract(self):
        """Subtract two numbers and update class variables
        """
        num1, num2 = self.get_input()
        self.num_calculations += 1
        self.previous_result = num1 - num2
        self.total_results += self.previous_result

        self.menu()


    def multiply(self):
        """Multiply two numbers and update class variables
        """
        num1, num2 = self.get_input()
        self.num_calculations += 1
        self.previous_result = num1 * num2
        self.total_results += self.previous_result

        self.menu()

    
    def divide(self):
        """Divide two numbers and update class variables
        """
        num1, num2 = self.get_input()
        self.num_calculations += 1
        self.previous_result = num1 / num2
        self.total_results += self.previous_result

        self.menu()

    
    def exponent(self):
        """Exponentiate two numbers and update class variables
        """
        num1, num2 = self.get_input()
        self.num_calculations += 1
        self.previous_result = num1 ** num2
        self.total_results += self.previous_result

        self.menu()


    def log(self):
        """Take the logarithm two numbers and update class variables
        """
        num1, num2 = self.get_input()
        self.num_calculations += 1
        self.previous_result = math.log(num2, num1)
        self.total_results += self.previous_result

        self.menu()


def main():
    calculator = Calculator()
    calculator.menu()

if __name__ == "__main__":
    main()