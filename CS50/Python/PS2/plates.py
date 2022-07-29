def main():
    plate = input("Plate: ")
    if is_valid(plate):
        print("Valid")
    else:
        print("Invalid")


def is_valid(s):
    if len(s) < 2 or len(s) > 6:
        return False
    elif not s[:2].isalpha():
        return False

    num_found = False
    for char in s[2:]:
        if not char.isalnum():
            return False
        elif char.isnumeric():
            if not num_found and char == '0':
                return False
            num_found = True
        else:
            if num_found:
                return False

    return True

main()