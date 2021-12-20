from crossword import *

domains = {Variable(1, 4, 'down', 4): {'TEN', 'TWO', 'FOUR', 'ONE', 'SIX', 'SEVEN', 'EIGHT', 'THREE', 'NINE', 'FIVE'},
           Variable(0, 1, 'down', 5): {'TEN', 'TWO', 'FOUR', 'ONE', 'SIX', 'SEVEN', 'EIGHT', 'THREE', 'NINE', 'FIVE'},
           Variable(0, 1, 'across', 3): {'TEN', 'TWO', 'FOUR', 'ONE', 'SIX', 'SEVEN', 'EIGHT', 'THREE', 'NINE', 'FIVE'},
           Variable(4, 1, 'across', 4): {'TEN', 'TWO', 'FOUR', 'ONE', 'SIX', 'SEVEN', 'EIGHT', 'THREE', 'NINE', 'FIVE'}
           }

for variable in domains:
    for word in variable.words:
        if len(word) != variable.length:
            domains[variable].remove(word)

print(domains)