import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count


    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count


    def __str__(self):
        return f"{self.cells} = {self.count}"


    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        # Check if every cell is mine
        if len(self.cells) == self.count:
            return set(self.cells)
        else:
            return set()


    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        # Check if no cells are mines
        if self.count == 0:
            return set(self.cells)
        else:
            return set()


    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        # Check if cell still in cells
        if cell in self.cells:
            # If so, remove and lower mine count
            self.cells.remove(cell)
            self.count -= 1


    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        # Check if cell still in cells
        if cell in self.cells:
            # If so, remove
            self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width
        self.poss_moves = set()
        for row in range(self.height):
            for colm in range(self.width):
                self.poss_moves.add((row, colm))

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)


    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)


    def find_neighbors(self, cell):
        count = 0
        i,j = cell
        neighbors = set()
        # Check row/colm pairs to see if any are mines, if yes, add to count
        for row in range(i-1, i+2):
            if row < 0 or row > self.height:
                continue
            for colm in range(j-1, j+2):
                if colm < 0 or colm > self.width:
                    continue
                if (row, colm) in self.mines:
                    count += 1
                elif (row, colm) not in self.safes:
                    neighbors.add((row, colm))

        return neighbors, count


    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        # Mark cell as move that has been made
        self.moves_made.add(cell)

        # Mark cell as safe
        self.safes.add(cell)

        # Get neighbors
        neighbors, mine_count = self.find_neighbors(cell)
        count -= mine_count

        # Make new sentance and add to knowledge if new
        new_sent = Sentence(neighbors, count)
        if new_sent not in self.knowledge:
            self.knowledge.append(new_sent)

        def knowledge_cycle(knowledge):
            new_knowledge = []
            # Go through knowledge and see if we figured out anything new
            for sent in knowledge.copy():
                # Check mines
                known_mine_cells = sent.known_mines()
                for known_mine in known_mine_cells:
                    self.mark_mine(known_mine)
                
                # Check safes
                known_safe_cells = sent.known_safes()
                for known_safe in known_safe_cells:
                    self.mark_safe(known_safe)

            # Infer stuff
            for sent1 in knowledge.copy():
                # Check for solved sentances
                if sent1.count == len(sent1.cells):
                    for cell in sent1.cells.copy():
                        self.mark_mine(cell)
                elif sent1.count == 0:
                    for cell in sent1.cells.copy():
                        self.mark_safe(cell)

                # Compare two sentances
                for sent2 in knowledge:
                    if (sent1 == sent2 or
                        len(sent1.cells) == 0 or
                        len(sent2.cells) == 0
                        ):
                        continue
                    
                    # Check for subsets
                    if sent1.cells.issubset(sent2.cells):
                        new_sent = Sentence(sent2.cells - sent1.cells, sent2.count - sent1.count)
                        if new_sent not in knowledge:
                            new_knowledge.append(new_sent)
                    elif sent2.cells.issubset(sent1.cells):
                        new_sent = Sentence(sent1.cells - sent2.cells, sent1.count - sent2.count)
                        if new_sent not in knowledge:
                            new_knowledge.append(new_sent)

            return new_knowledge


        while True:
            new_knowledge = knowledge_cycle(self.knowledge)
            for wisdom in new_knowledge:
                self.knowledge.append(wisdom)
            if len(new_knowledge) == 0:
                break


    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        # Find available moves and see if any are safe
        avail_moves = self.poss_moves - self.moves_made
        for move in avail_moves:
            if move in self.safes:
                return move

        # Return None if no safe moves found
        return None

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        avail_moves = list(self.poss_moves - self.moves_made)
        random.shuffle(avail_moves)
        
        # Return None if no avail moves
        if len(avail_moves) == 0:
            return None

        # Ensure random move is not mine
        for move in avail_moves:
            if move not in self.mines:
                return move
