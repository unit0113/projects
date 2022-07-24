normal_words = input("Give me a string with smilies please: ")
smilie_words = normal_words.replace(':)', '🙂').replace(':(', '🙁')
print(smilie_words)