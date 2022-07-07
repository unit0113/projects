with open(r'AoC\2021\Day_15\input.txt', 'r') as file:
    risks = [[int(risk) for risk in list(line.strip())] for line in file.readlines()]
    

def find_safest_path(risks):
    path_risks = []


    def find_safest_path_helper(risks, start, current_risk, current_path):
        x_start, y_start = start
        for move in valid_moves(risks, start, current_path):
            new_x, new_y = move
            new_risk = current_risk + risks[new_x][new_y]
            if new_x != len(risks[0]) - 1 or new_y != len(risks) - 1:
                new_path = current_path[:]
                new_path.append(move)
                find_safest_path_helper(risks, move, new_risk, new_path)
            else:
                path_risks.append(new_risk)


    find_safest_path_helper(risks, (0,0), 0, [])

    return min(path_risks)


def valid_moves(risks, start, current_path):
    moves = []
    x_start, y_start = start
    '''if x_start > 0:
        move = (x_start-1, y_start)
        if move not in current_path:
            moves.append(move)'''
    if x_start < len(risks[0]) - 1:
        move = (x_start+1, y_start)
        if move not in current_path:
            moves.append(move)
    '''if y_start > 0:
        move = (x_start, y_start-1)
        if move not in current_path:
            moves.append(move)'''
    if y_start < len(risks) - 1:
        move = (x_start, y_start+1)
        if move not in current_path:
            moves.append(move)

    return moves


print(find_safest_path(risks))