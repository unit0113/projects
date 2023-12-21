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

    if new_board[action[0]][action[1]] == EMPTY:
        # Set cell to current player
        new_board[action[0]][action[1]] = player(board)

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
                return player

    # Transpose board to get each column in own list
    columns = list(map(list, zip(*board)))

    # check columns
    for colm in columns:
        for player in players:
            if colm.count(player) == 3:
                return player

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

    results = []

    # find the max tree
    def find_max(board, alpha, beta):
        if terminal(board):
            return utility(board)
        max_eval = -math.inf
        for move in actions(board):
            value = find_min(result(board, move), alpha, beta)
            max_eval = max(max_eval, value)
            alpha = max(alpha, value)
            if beta <= alpha:
                break

        return max_eval


    # find the min tree
    def find_min(board, alpha, beta):
        if terminal(board):
            return utility(board)
        min_eval = math.inf
        for move in actions(board):
            value = find_max(result(board, move), alpha, beta)
            min_eval = min(min_eval, value)
            beta = min(beta, value)
            if beta <= alpha:
                break

        return min_eval

    # Actually run the algo
    if player(board) == X:
        for move in actions(board):
            results.append([find_min(result(board, move), alpha, beta), move])
        return sorted(results, key=lambda x: x[0], reverse=True)[0][1]

    # run for O player
    else:
        for move in actions(board):
            results.append([find_max(result(board, move), alpha, beta), move])
        return sorted(results, key=lambda x: x[0])[0][1]
