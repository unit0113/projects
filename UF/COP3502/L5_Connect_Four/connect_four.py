TOKENS = [None, 'x', 'o']


def initialize_board(array):
    """ Resets all spaces on already created board to '-'

    Args:
        array (list): 2D list representation of game board 
    """
    for row_index, row in enumerate(array):
        for col_index, _ in enumerate(row):
            array[row_index][col_index] = '-'


def create_board(rows, columns):
    """Creates game board

    Args:
        rows (int): Number of rows
        columns (int): Number of Columns

    Returns:
        List: 2D list representation of game board
    """

    # Using * columns because for in range... results in reference errors
    return [['-'] * columns for _ in range(rows)]


def print_board(board):
    """Prints game board
    Args:
        board (List): 2D list representation of game board
    """

    # reversed instead of .reverse as .reverse in loop initialization results in NoneTypeError
    for row in reversed(board):
        print(' '.join(row))
    
    print()


def insert_chip(board, col, chip_type):
    """Insert chip into game board

    Args:
        board (list): 2D list representation of game board
        col (int): Column to play chip in
        chip_type (char): Player token

    Returns:
        int: Row that chip was played in
    """

    # Find first empty row
    row = 0
    while board[row][col] != '-':
        row += 1

    board[row][col] = chip_type

    return row

def check_if_winner(board, col, row, chip_type):
    """Check if last play resulted in a winner, print winner if found

    Args:
        board (list): 2D list representation of game board
        col (int): Column number of last play
        row (int): Row number of last play
        chip_type (char): Player token of last play

    Returns:
        bool: If there is a winner
    """

    # Check rows, find furthest left of same type
    start = end = col
    while start > 0 and board[row][start - 1] == chip_type:
        start -= 1

    # Find end of run
    while end < len(board[0]) - 1 and board[row][end + 1] == chip_type:
        end += 1

    # Print winner if there is one
    if end - start >= 3:
        print(f'Player {TOKENS.index(chip_type)} won the game!')
        return True

    # Check cols
    start = row
    while start >= 0 and board[start - 1][col] == chip_type:
        start -= 1

    # Find bottom
    end = row
    while end < len(board) - 1 and board[end + 1][col] == chip_type:
        end += 1

    # Print winner if there is one
    if end - start >= 3:
        print(f'Player {TOKENS.index(chip_type)} won the game!')
        return True

    # No winner
    return False


def check_if_draw(board):
    """Check if game ends in draw

    Args:
        board (list): 2D list representation of game board

    Returns:
        bool: If game ends in draw
    """
    if sum([row.count('-') for row in board]) == 0:
        print('Draw. Nobody wins.')
        return True
    
    return False


def main():
    rows = int(input('What would you like the height of the board to be? '))
    columns = int(input('What would you like the length of the board to be? '))

    board = create_board(rows, columns)
    print_board(board)
    
    print('Player 1: x')
    print('Player 2: o')
    print()

    player = 1
    run = True
    while run:
        selection = int(input(f'Player {player}: Which column would you like to choose? '))
        row = insert_chip(board, selection, TOKENS[player])
        print_board(board)

        if check_if_winner(board, selection ,row, TOKENS[player]) or check_if_draw(board):
            break

        player = 1 if player == 2 else 2

if __name__ == "__main__":
    main()