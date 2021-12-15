import random


corpus = {"1.html": {"2.html", "3.html"}, "2.html": {"3.html"}, "3.html": {"2.html"}}
damping_factor = 0.85
page = "2.html"

# Initialize return dict
result = {}
for site in corpus:
    result[site] = 0

# Add values for not damped
if len(corpus[page]) == 0:
    i = 1 / len(corpus)
    for site in corpus:
        result[site] = round(result[site] + i, 10)
else:
    # Add values for random page (damping)
    damped = round((1 - damping_factor) / len(corpus), 10)
    for site in result:
        result[site] += damped

    # Add values for not damped
    i = round(damping_factor / len(corpus[page]), 10)
    for site in corpus[page]:
        result[site] = round(result[site] + i, 10)

#print(result)

# Initialize return dict, error min, and constant for formula
N = len(corpus)
page_rank = {site: 1 / N for site in corpus}
error = 0.001
constant = round((1 - damping_factor) / N, 10)

# Run through loop
while True:
    old_page_rank = page_rank.copy()
    for site in corpus:
        sum = 0
        for i in corpus:
            # Check if no links
            if len(corpus[i]) == 0:
                sum += page_rank[i] / N
            # If there are links
            elif site in corpus[i]:
                sum += page_rank[i] / len(corpus[i])

        # use equation
        page_rank[site] = constant + (damping_factor * sum)

    # Check error values
    error_check = True
    for site in corpus:
        if abs(page_rank[site] - old_page_rank[site]) > error:
            error_check = False
            break
    
    # Return page rank if all errors less than error value
    if error_check:
        #return page_rank
        print(page_rank)
        break

