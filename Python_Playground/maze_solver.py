import curses
from curses import wrapper
from queue import Queue
import time


MAZE = [
    ['#','#','#','#','#','O','#','#','#'],
    ['#',' ',' ',' ',' ',' ',' ',' ','#'],
    ['#',' ','#','#',' ','#','#',' ','#'],
    ['#',' ','#',' ',' ',' ','#',' ','#'],
    ['#',' ','#',' ','#',' ','#',' ','#'],
    ['#',' ','#',' ','#',' ','#',' ','#'],
    ['#',' ','#',' ','#',' ','#','#','#'],
    ['#',' ',' ',' ',' ',' ',' ',' ','#'],
    ['#','#','#','#','#','#','#','X','#']
]


def find_path(maze, stdscr):
    start_symbol = 'O'
    end_symbol = 'X'
    start = find_point(maze, start_symbol)
    end = find_point(maze, end_symbol)
    q = Queue()
    q.put((start, [start]))
    explored = set(start)

    while q:
        current, path = q.get()
        row, col = current

        stdscr.clear()
        print_maze(maze, stdscr, path)
        stdscr.refresh()
        time.sleep(0.2)

        if current == end:
            return path

        neighbors = find_valid_neighbors(maze, row, col)
        for neighbor in [neighbor for neighbor in neighbors if neighbor not in explored]:
            new_path = path + [neighbor]
            q.put((neighbor, new_path))
            explored.add(neighbor)

        
def find_valid_neighbors(maze, row, col):
    neighbors = []

    if row > 0 and maze[row-1][col] != '#':
        neighbors.append((row - 1, col))

    if row < len(maze) and maze[row+1][col] != '#':
        neighbors.append((row + 1, col))

    if col > 0 and maze[row][col-1] != '#':
        neighbors.append((row, col - 1))

    if col < len(maze[0]) and maze[row][col+1] != '#':
        neighbors.append((row, col + 1))

    return neighbors


def find_point(maze, symbol):
    for row_index, row in enumerate(maze):
        for col_index, payload in enumerate(row):
            if payload == symbol:
                return (row_index, col_index)

    return None

def print_maze(maze, stdscr, path=[]):
    BLUE = curses.color_pair(1)
    RED = curses.color_pair(2)

    for row_index, row in enumerate(maze):
        for col_index, payload in enumerate(row):
            if (row_index, col_index) in path:
                stdscr.addstr(row_index, col_index*2, 'X', RED)
            else:
                stdscr.addstr(row_index, col_index*2, payload, BLUE)


def main(stdscr):
    curses.init_pair(1, curses.COLOR_BLUE, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)

    find_path(MAZE, stdscr)
    stdscr.getch()

wrapper(main)

