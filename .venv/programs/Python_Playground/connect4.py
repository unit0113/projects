import pygame
pygame.init()
import math
import copy
from multiprocessing import Pool
import os

# Initialize the main window
screen_height = pygame.display.get_desktop_sizes()[0][1]
WIDTH = round(screen_height * 0.9)
HEIGHT = round(screen_height * 0.9) * 6 // 7
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT+100))
pygame.display.set_caption("Connect 4")

# Colors
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

FPS = 60
HUMAN = 'Human'
AI = 'Computer'
HUMAN_COLOR = RED
AI_COLOR = YELLOW

ROW_COUNT = 6
COLUMN_COUNT = 7
WIN_LENGTH = 4
MINIMAX_DEPTH = 3


class Tile:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.x = col * WIDTH // COLUMN_COUNT 
        self.y = row * HEIGHT // ROW_COUNT + 100
        self.color = WHITE
        self.dimensions = WIDTH // COLUMN_COUNT
        self.rect = pygame.Rect(self.x, self.y, self.dimensions, self.dimensions)


    def draw(self, window):
        pygame.draw.circle(window, self.color, (self.x + self.dimensions // 2, self.y + self.dimensions // 2), self.dimensions // 2 - self.dimensions // 10)


class Grid:
    def __init__(self):
        self.open_spaces = [ROW_COUNT-1] * COLUMN_COUNT
        self.active_player = HUMAN
        self.last_play = (None, None)
        self.grid = []
        self.check_grid = []
        for i in range(ROW_COUNT):
            self.grid.append([])
            self.check_grid.append([0] * COLUMN_COUNT)
            for j in range(COLUMN_COUNT):
                tile = Tile(i, j)
                self.grid[i].append(tile)
            
    def __next_player(self):
        if self.active_player == AI:
            self.active_player = HUMAN
        else:
            self.active_player = AI

    
    def get_valid_moves(self):
        return [col for col, row in enumerate(self.open_spaces) if self.open_spaces[col] != -1]


    def play(self, active_col, skip_endgame = False):
        row = self.open_spaces[active_col]

        # Check if row is full
        if row < 0:
            return False

        self.grid[row][active_col].color = RED if self.active_player == HUMAN else YELLOW
        self.check_grid[row][active_col] = 1 if self.active_player == HUMAN else -1
        self.open_spaces[active_col] -= 1
        self.last_play = (row, active_col)

        if not skip_endgame:
            if self.game_over():
                endgame(self.active_player)
            elif self.game_over() == 0:
                endgame(None)

        self.__next_player()        

        return True


    def game_over(self):
        if self.check_win():
            if self.active_player == HUMAN:
                return 100_000
            else:
                return -100_000

        elif sum(self.open_spaces) == -COLUMN_COUNT:
            return 0

        return None


    def __check_win_horz(self, player_color):
        row, col = self.last_play
        # Find furthest left of same color
        start = col
        while start > 0 and self.grid[row][start-1].color == player_color:
            start -= 1

        # Find furthest right
        end = col
        while end < 6 and self.grid[row][end+1].color == player_color:
            end += 1    
        
        if end - start >= 3:
            return True

    
    def __check_win_vert(self, player_color):
        row, col = self.last_play
        # Find top
        start = row
        while start >= 0 and self.grid[start-1][col].color == player_color:
            start -= 1

        # Find bottom
        end = row
        while end < 5 and self.grid[end+1][col].color == player_color:
            end += 1

        if end - start >= 3:
            return True


    def __check_win_diag(self, player_color):
        row, col = self.last_play
        # Down-right
        start_row = row
        start_col = col
        while start_row > 0 and start_col > 0 and self.grid[start_row-1][start_col-1].color == player_color:
            start_row -= 1
            start_col -= 1

        end_row = row
        end_col = col
        while end_row < 5 and end_col < 6 and self.grid[end_row+1][end_col+1].color == player_color:
            end_row += 1
            end_col += 1

        if end_row - start_row >= 3:
            return True

        # Up-right
        start_row = row
        start_col = col
        while start_row < 5 and start_col > 0 and self.grid[start_row+1][start_col-1].color == player_color:
            start_row += 1
            start_col -= 1

        end_row = row
        end_col = col
        while end_row > 0 and end_col < 6 and self.grid[end_row-1][end_col+1].color == player_color:
            end_row -= 1
            end_col += 1        

        if start_row - end_row >= 3:
            return True


    def check_win(self):
        row, col = self.last_play
        player_color = self.grid[row][col].color

        if (self.__check_win_horz(player_color) or
            self.__check_win_vert(player_color) or
            self.__check_win_diag(player_color)
            ):
            return True

        else:
            return False


    def draw(self, window):
        # Draw top bar
        pygame.draw.rect(window, GREY, (0, 0, WIDTH, 100))

        # Draw circles
        for row in self.grid:
            for tile in row:
                tile.draw(window)


class Active:
    def __init__(self):
        self.active_col = 0
        self.x = WIDTH // 21
        self.y = 10


    def calc_x(self):
        self.x = WIDTH // 21 + self.active_col * WIDTH // 7

    
    def move_left(self):
        if self.active_col != 0:
            self.active_col -= 1
        else:
            self.active_col = 6

        self.calc_x()


    def move_right(self):
        if self.active_col != 6:
            self.active_col += 1
        else:
            self.active_col = 0

        self.calc_x()


    def draw(self, window):
        pygame.draw.polygon(window, GREEN, ((self.x, self.y),(self.x,self.y+40),(self.x-30,self.y+40), (self.x+30, self.y+80), (self.x+90,self.y+40), (self.x+60,self.y+40), (self.x+60, self.y)))



def result(grid, col):
    """
    Returns the grid that results from making placing puck into specific column.
    """
    # Initialize deep copy of board and row and column indecies
    new_grid = copy.deepcopy(grid)

    new_grid.play(col, skip_endgame='True')

    return new_grid


def evaluate_slice(slice, piece):
    score = 0
    opp_piece = -piece

    if slice.count(piece) == 3 and slice.count(0) == 1:
        score += 5
    elif slice.count(piece) == 2 and slice.count(0) == 2:
        score += 2

    if slice.count(opp_piece) == 3 and slice.count(0) == 1:
        score -= 4

    return score


def score_position(grid):
    piece = 1 if grid.active_player == HUMAN else -1
    score = 0

    ## Score center column
    center_array = [row[COLUMN_COUNT//2] for row in grid.check_grid]
    center_count = center_array.count(piece)
    score += center_count * 3

    ## Score Horizontal
    for row in grid.check_grid:
        for col in range(COLUMN_COUNT-3):
            window = row[col:col+WIN_LENGTH]
            score += evaluate_slice(window, piece)

    ## Score Vertical
    columns_wise_grid = list(map(list, zip(*grid.check_grid)))
    for col in columns_wise_grid:
        for row in range(ROW_COUNT-3):
            window = col[row:row+WIN_LENGTH]
            score += evaluate_slice(window, piece)

    ## Score posiive sloped diagonal
    for row in range(ROW_COUNT-3):
        for col in range(COLUMN_COUNT-3):
            window = [grid.check_grid[row+i][col+i] for i in range(WIN_LENGTH)]
            score += evaluate_slice(window, piece)

    for row in range(ROW_COUNT-3):
        for col in range(COLUMN_COUNT-3):
            window = [grid.check_grid[row+3-i][col+i] for i in range(WIN_LENGTH)]
            score += evaluate_slice(window, piece)

    return score


def minimax(grid, alpha, beta, depth):
    """
    Returns the optimal action for the current player on the board.
    """

    results = []

    # find the max tree
    def find_max(grid, alpha, beta, depth):
        game_over_result = grid.game_over()
        if game_over_result == 0 or depth == 0:
            return score_position(grid)
        elif game_over_result != None:
            return game_over_result
        max_eval = -math.inf
        for move in grid.get_valid_moves():
            value = find_min(result(grid, move), alpha, beta, depth-1)
            max_eval = max(max_eval, value)
            alpha = max(alpha, value)
            if beta <= alpha:
                break

        return max_eval


    # find the min tree
    def find_min(grid, alpha, beta, depth):
        game_over_result = grid.game_over()
        if game_over_result == 0 or depth == 0:
            return score_position(grid)
        elif game_over_result != None:
            return game_over_result
        min_eval = math.inf
        for move in grid.get_valid_moves():
            value = find_max(result(grid, move), alpha, beta, depth-1)
            min_eval = min(min_eval, value)
            beta = min(beta, value)
            if beta <= alpha:
                break

        return min_eval

    # Run minimax
    for move in grid.get_valid_moves():
        results.append([find_min(result(grid, move), alpha, beta, depth), move])
    return sorted(results, key=lambda x: x[0], reverse=True)[0][1]


def AI_play(grid):
    move = minimax(grid, -math.inf, math.inf, MINIMAX_DEPTH)
    grid.play(move)


def draw(window, grid, active):
    window.fill(BLACK)
    grid.draw(window)
    active.draw(window)
    pygame.display.update()


def endgame(player):
    if player:
        print(f'The {player} wins!')
    else:
        print('Draw!')


def main():
    clock = pygame.time.Clock()
    grid = Grid()
    active = Active()

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    active.move_left()
                elif event.key == pygame.K_RIGHT:
                    active.move_right()
                elif event.key == pygame.K_SPACE:
                    if grid.play(active.active_col):
                        draw(WINDOW, grid, active)
                        AI_play(grid)

        draw(WINDOW, grid, active)
        

if __name__ == "__main__":
    main()