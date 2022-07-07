from cs50 import get_string
import sys


def invalid():
    print("INVALID")
    sys.exit()


def luhn(strcc):

    # Check valid length
    if len(strcc) not in [13, 15, 16]:
        invalid()

    # Check valid leading numbers
    # Check AMEX
    if len(strcc) == 15:
        if not int(strcc[0]) == 3:
            invalid()
        else:
            if int(strcc[1]) not in [4, 7]:
                invalid()
            else:
                flag = "AMEX"

    # Check Visa
    elif int(strcc[0]) == 4:
        if len(strcc) not in [13, 16]:
            invalid()
        else:
            flag = "VISA"

    # Check MA
    elif len(strcc) == 16:
        if not int(strcc[0]) == 5:
            invalid()
        else:
            if not int(strcc[1]) in range(1, 6):
                invalid()
            else:
                flag = "MASTERCARD"

    # First pass
    sum = 0
    for i in range(2, len(strcc) + 1, 2):
        num = 2 * int(strcc[-i])
        # Check if product is double digit
        if num > 9:
            num = str(num)
            num1 = int(num[0])
            num2 = int(num[1])
            num = num1 + num2

        # Add to checksum
        sum += num

    # Second pass
    for i in range(1, len(strcc) + 1, 2):
        sum += int(strcc[-i])

    # Print result
    if sum % 10 == 0:
        print(flag)

    else:
        invalid()


def main():
    # Get credit card number
    cc = -1
    while cc < 1:
        strcc = get_string("Number: ")
        cc = int(strcc)

    luhn(strcc)


main()
