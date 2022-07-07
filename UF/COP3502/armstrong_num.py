def is_armstrong(number):
    number_list = [int(num) for num in list(str(number))]
    length = len(number_list)
    sum = 0
    for num in number_list:
        sum += num ** length

    return sum == number


start, end = input().split()

for num in range(int(start), int(end)+1):
    if is_armstrong(num):
        print(num)