with open('input.txt', 'r') as file:
    selections = [elem.strip() for elem in file.readline().split(',')]
    next(file)

    boards = []
    board = []
    for line in file:
        if not line.strip():
            boards.append(board)
            board = []
            continue
        
        board_row = [elem.strip() for elem in line.split(' ') if elem.strip()]
        board.append(board_row)


# Part 1
def check_winner(boards):
    for board_number, board in enumerate(boards):
        for row in board:
            if row.count('X') == 5:
                return board_number

        columns = list(map(list, zip(*board)))
        for colm in columns:
            if colm.count('X') == 5:
                return board_number
    
    return False


for selection in selections:
    for board in boards:
        for row in board:
            for i, item in enumerate(row):
                if item == selection:
                    row[i] = 'X'

    winner = check_winner(boards)
    if winner:
        break

board_sum = 0
for row in boards[winner]:
    board_sum += sum([int(elem) for elem in row if elem != 'X'])

answer = board_sum * int(selection)
#print(answer)


# Part 2
def check_board(board):
    for row in board:
        if row.count('X') == 5:
            return True

    columns = list(map(list, zip(*board)))
    for colm in columns:
        if colm.count('X') == 5:
            return True
    
    return False


winner_list = []

for selection in selections:
    for board_index, board in enumerate(boards):
        if board_index not in winner_list:
            for row in board:
                for i, item in enumerate(row):
                    if item == selection:
                        row[i] = 'X'

            if check_board(board):
                winner_list.append(board_index)
    if len(winner_list) == len(boards):
        break

loser = winner_list[-1]

board_sum = 0
for row in boards[loser]:
    board_sum += sum([int(elem) for elem in row if elem != 'X'])

answer = board_sum * int(selection)
print(answer)