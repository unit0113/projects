variable = input("Enter a variable name: ")

word = ''
words = []
for char in variable:
    if char.isupper() and word:
        words.append(word)
        word = char.lower()
    else:
        word += char

words.append(word)

print('_'.join(words))