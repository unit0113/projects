def main():
    num1 = 'a'
    num2 = 'b'

    while not num1.isnumeric() or not num2.isnumeric():
        raw_input = input("Fraction: ")
        try:
            percentage = convert(raw_input)
        except ValueError:
            continue
        except ZeroDivisionError:
            num2 = 'b'
            continue

        break

    print(gauge(percentage))


def convert(fraction):
    num1, num2 = fraction.split('/')

    try:
        num1 = int(num1)
        num2 = int(num2)

    except ValueError:
        raise ValueError
    if num2 == 0:
        raise ZeroDivisionError
    elif num1 > num2:
        raise ValueError

    return round(100 * num1 / num2)


def gauge(percentage):
    if percentage >= 99:
        return 'F'
    elif percentage <= 1:
        return 'E'
    else:
        return f'{percentage}%'


if __name__ == "__main__":
    main()