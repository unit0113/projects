vowels = 'aeiou'

str_in = input("Input: ")
str_out = ''
for char in str_in:
    if char.lower() in vowels:
        continue
    str_out += char

print(f'Output: {str_out}')