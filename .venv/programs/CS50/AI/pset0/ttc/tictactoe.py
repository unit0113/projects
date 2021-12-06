"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # count all the EMPTY's
    count = sum(sub_list.count(EMPTY) for sub_list in board)

    # if event number of empty's, return O, else X
    if count % 2 == 0:
        return O
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """

    # initial set of available moves
    moves = set()

    # loop through all the cells
    for i in range(3):
        for j in range(3):
            # Check if empty in cell
            if board[i][j] == EMPTY:
                moves.add((i, j))

    return moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # Initialize deep copy of board and row and column indecies
    new_board = copy.deepcopy(board)
    x, y = action

    # Check if board is legal
    if new_board[x][y] != EMPTY:
        raise InvalidMoveError

    else:
        # Set cell to current player
        new_board[x][y] = player(board)

    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    players = [X, O]

    # check diagonals
    for player in players:
        if (player == board[0][0] == board[1][1] == board[2][2] or
            player == board[0][2] == board[1][1] == board[2][0]
            ):
            return player

    # check rows
    for row in board:
        for player in players:
            if row.count(player) == 3:
                print(player)

    # Transpose board to get each column in own list
    columns = list(map(list, zip(*board)))

    # check columns
    for colm in columns:
        for player in players:
            if colm.count(player) == 3:
                print(player)

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # Check if there is a winner or if there are no available moves
    if winner(board) or len(actions(board)) == 0:
        return True
    else:
        return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """

    victor = winner(board)

    if victor == X:
        return 1
    elif victor == O:
        return -1
    else:
        return 0


def minimax(board, alpha, beta):
    """
    Returns the optimal action for the current player on the board.
    """

    # Check if terminal and return value if true
    if terminal(board):
        return utility(board)

    # for X player
    if player(board) == X:
        max_eval = -math.inf
        for move in actions(board):
            value = minimax(result(board, move), alpha, beta)
            max_eval = max(max_eval, value)
            alpha = max(alpha, value)
            if beta <= alpha:
                break

        return max_eval

    # for O player
    else:
        min_eval = math.inf
        for move in actions(board):
            value = minimax(result(board, move), alpha, beta)
            min_eval = min(max_eval, value)
            beta = min(beta, value)
            if beta <= alpha:
                break

        return min_eval