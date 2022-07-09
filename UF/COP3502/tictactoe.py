# required methods
def print_board(board):
    for row in board:
        print(' '.join(row))
        
    print()


def initialize_board():
    return [['-'] * 3 for _ in range(3)]


def check_if_winner(board, chip_type):
    # Check row
    for row in board:
        if row.count(chip_type) == 3:
            return True
    
    # Check cols
    for col in range(len(board[0])):
        if board[0][col] == board[1][col] and board[1][col] == board[2][col]:
            return True

    # Check diags
    if (board[0][0] == chip_type
        and board[1][1] == chip_type
        and board[2][2] == chip_type
        or board[0][2] == chip_type
        and board[1][1] == chip_type
        and board[2][0] == chip_type):
        return True 

    return False


def board_is_full(board):
    if sum([row.count('-') for row in board]) == 0:
        print("It's a tie!")
        return True
        
    return False


def available_square(board, row, col):
    return board[row][col] == '-'


def mark_square(board, row, col, player):
    board[row][col] = player


# Personal solution
TOKENS = [None, 'x', 'o']

class Board:
    def __init__(self):
        self.board = [['-'] * 3 for _ in range(3)]
        self.last_play = (None, None)

    def print_board(self):
        for row in self.board:
            print(' '.join(row))
        
        print()

    def check_if_draw(self):
        if sum([row.count('-') for row in self.board]) == 0:
            print("It's a tie!")
            return True
        
        return False

    def check_if_winner(self, player):
        # Check row
        if self.board[self.last_play[0]].count(TOKENS[player]) == 3:
            print(f'Player {player} has won!')
            return True
        
        # Check cols
        elif (self.board[0][self.last_play[1]] == TOKENS[player]
              and self.board[1][self.last_play[1]] == TOKENS[player]
              and self.board[2][self.last_play[1]] == TOKENS[player]):
            print(f'Player {player} has won!')
            return True

        # Check diags
        elif (self.board[0][0] == TOKENS[player]
              and self.board[1][1] == TOKENS[player]
              and self.board[2][2] == TOKENS[player]
              or self.board[0][2] == TOKENS[player]
              and self.board[1][1] == TOKENS[player]
              and self.board[2][0] == TOKENS[player]):
            print(f'Player {player} has won!')
            return True 

        return False

    def play_turn(self, player):
        row = int(input('Enter a row number (0, 1, or 2): '))
        col = int(input('Enter a column number (0, 1, or 2): '))
        if row > 2 or col > 2 or row < 0 or col < 0:
            print('This position is off the bounds of the board! Try again.')
            return False
        
        if self.board[row][col] != '-':
            print('Someone has already made a move at this position! Try again.')
            return False

        self.board[row][col] = TOKENS[player]
        self.last_play = (row, col)

        self.print_board()

        return True


def main():
    board = Board()
    
    print('Player 1: x')
    print('Player 2: o')
    print()

    board.print_board()

    player = 1
    run = True
    while run:
        print(f"Player {player}'s Turn ({TOKENS[player]}):")

        valid_turn = False
        while not valid_turn:
            valid_turn = board.play_turn(player)

        if board.check_if_winner(player) or board.check_if_draw():
            break

        player = 1 if player == 2 else 2


if __name__ == "__main__":
    main()
