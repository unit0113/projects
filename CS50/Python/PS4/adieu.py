inputs = ['Adieu, adieu, to ']

while True:
    try:
        inputs.append(input())
    except EOFError:
        break

if len(inputs) == 2:
    print(inputs[0] + inputs[1])
elif len(inputs) == 3:
    print(inputs[0] + inputs[1] + ' and ' + inputs[-1])
else:
    print(inputs[0] + inputs[1] +  ', ' + ', '.join(inputs[2:-1]) + ', and ' + inputs[-1])
