from cs50 import get_string


def ColemanLiau(L, S):
    grade = round(0.0588 * L - 0.296 * S - 15.8)
    return grade


def printgrade(grade):
    if grade > 16:
        print("Grade 16+")
    elif grade < 1:
        print("Before Grade 1")
    else:
        print("Grade {}" .format(grade))


def main():

    # Get text
    text = get_string("Text: ")
    
    # Variables
    l = 0
    w = 1
    s = 0
    wc = False
    punc = [".", "?", "!"]
    
    # Loop through text
    for i in range(len(text)):
        
        # Count letters
        if text[i].isalpha() == True:
            l += 1
            wc = True
            
        # Count words
        elif wc == True and text[i].isspace() == True:
            w += 1
            wc = False
            
        # Count sentances
        elif text[i] in punc:
            s += 1
 
    # Math
    L = (l / w) * 100
    S = (s / w) * 100
    
    # Determine grade
    grade = ColemanLiau(L, S)
    
    # Print grade
    printgrade(grade)


main()