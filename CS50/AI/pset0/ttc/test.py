import copy
import tictactoe as ttt

X = "X"
O = "O"
EMPTY = None

board = [[EMPTY, EMPTY, EMPTY],
         [EMPTY, EMPTY, EMPTY],
         [EMPTY, EMPTY, EMPTY]]

count = sum(sub_list.count(EMPTY) for sub_list in board)



moves = set()
for i in range(3):
    for j in range(3):
        if board[i][j] == EMPTY:
            moves.add((i, j))



#player = X
action = (0, 0)
new_board = copy.deepcopy(board)
x, y = action

# Check if board is legal
if new_board[x][y] != EMPTY:
    raise InvalidMoveError

else:
    new_board[x][y] = ttt.player(board)

#return new_board
#print(new_board)


board = [[O, EMPTY, X],
         [EMPTY, X, EMPTY],
         [X, EMPTY, X]]

players = [X, O]
# Transpose board to get each column in own list
columns = list(map(list, zip(*board)))

# check rows
for row in board:
    for player in players:
        if row.count(player) == 3:
            #return player
            print(player)

# check columns
for colm in columns:
    for player in players:
        if colm.count(player) == 3:
            #return player
            print(player)

# check diagonals
for player in players:
    if (player == board[0][0] == board[1][1] == board[2][2] or
        player == board[0][2] == board[1][1] == board[2][0]
        ):
        #return player
        print(player)