import os
import random
import re
import sys


DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

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

    return result


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Initialize return dict
    sampled = {site: 0 for site in corpus}
    
    # Determine what page to start on
    page = random.choice(list(corpus.keys()))
    sampled[page] += 1

    # Cycle through samples
    for i in range(n-1):
        # Get prob distribution for selected page
        dist = transition_model(corpus, page, damping_factor)
        page = random.choices(list(dist.keys()), weights=dist.values())[0]

        # Update sample counter
        sampled[page] += 1

    # Calc final rankings
    page_rank = {}
    for page in sampled:
        page_rank[page] = sampled[page] / n

    return page_rank



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
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
            return page_rank


if __name__ == "__main__":
    main()
