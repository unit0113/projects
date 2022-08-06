grocery_dic = {}
while True:
    try:
        item = input().upper()
    except EOFError:
        break

    if item in grocery_dic:
        grocery_dic[item] += 1
    else:
        grocery_dic[item] = 1

for key in sorted(grocery_dic):
    print(f'{grocery_dic[key]} {key}')