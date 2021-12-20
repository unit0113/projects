import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        # loop through all variables, and then all available words
        for variable in self.crossword.variables:
            for word in self.crossword.words:
                # if word not right length, remove as option for that variable
                if len(word) != variable.length:
                    self.domains[variable].remove(word)


    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        # initialize values
        overlap = self.crossword.overlaps[x, y]
        remove_words = []

        # check if any revisions need to be made
        if overlap is None:
            return False
        else:
            a, b = overlap

        # loop through words in x
        for word_x in self.domains[x]:
            poss_overlap = False
            # loop through words in y and check if overlap is possible
            for word_y in self.domains[y]:
                if word_x != word_y and word_x[a] == word_y[b]:
                    poss_overlap = True
                    break

            # remove word if it doesn't work
            if not poss_overlap:
                remove_words.append(word_x)

        for word in remove_words:
            self.domains[x].remove(word)

        return len(remove_words) > 0


    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        
        # find arcs if not given
        if arcs is None:
            arcs = []
            for var1 in self.crossword.variables:
                for var2 in self.crossword.neighbors(var1):
                    arcs.append((var1, var2))

        # initialize queue
        while arcs:
            x, y = arcs.pop(0)

            # add neighbors if revised
            if self.revise(x, y):
                if len(self.domains[x]) == 0 or len(self.domains[y]):
                    return False
                neighbors = [neighbor for neighbor in self.crossword.neighbors(x) if neighbor is not y]
                # add if new values found
                if neighbors != []:
                    arcs.append(neighbors)

        return True


    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        # check if all vars in assignment
        for var in self.crossword.variables:
            if (var not in assignment.keys() or
                assignment[var] not in self.crossword.words
                ):
                return False

        return True


    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        for var1 in assignment:
            # check for word length
            for word1 in assignment[var1]:
                if var1.length != len(word1):
                    return False

            # check for only using words once
            for var2 in assignment:
                word2 = assignment[var2]
                if var1 != var2:
                    if word1 == word2:
                        return False

                    # check overlaps
                    overlap = self.crossword.overlaps[var1, var2]
                    if overlap is not None:
                        a, b = overlap
                        if word1[a] != word2[b]:
                            return False

        return True


    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """

        # iniatialize values
        constraints = dict()

        for word_var in self.domains[var]:
            constraint_count = 0
            # go through all unassigned neighbors
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    continue
                # Check for overlap and see how much gets ruled out
                i_overlap, j_overlap = self.crossword.overlaps[var, neighbor]
                for word_neighbor in self.domains[neighbor]:
                    if word_var[i_overlap] != word_neighbor[j_overlap]:
                        constraint_count += 1
            # add to dict
            constraints[word_var] = constraint_count

        # Sort results
        domain_val_result = [val for val in sorted(constraints, key=constraints.get)]
        return domain_val_result           


    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """

        # Initialize values
        min_val = dict()

        for var in self.crossword.variables:
            # skip vars with answer
            if var in assignment:
                continue
            
            # add size of domains to the min_val dict
            value = len(self.domains[var])
            min_val[var] = value

        # find lowest values
        min_value = min(min_val.values())
        min_val_result = [var for var, val in min_val.items() if val == min_value]

        # return if only one option
        if len(min_val_result) == 1:
            return min_val_result[0]

        # add size of degree to the dict
        max_degree = dict()
        for var in min_val_result:
            degree = len(self.crossword.neighbors(var))
            max_degree[var] = degree

        # find highest degree
        max_value = max(max_degree.values())
        max_degree_result = [var for var, val in max_degree.items() if val == max_value]

        # return answer
        return max_degree_result[0]


    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        
        # return assignment if complete
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)
        for word in self.order_domain_values(var, assignment):
            assignment[var] = word
            if self.consistent(assignment):
                result = self.backtrack(assignment)
                if result is None:
                    assignment[var] = None
                else:
                    return result

        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
