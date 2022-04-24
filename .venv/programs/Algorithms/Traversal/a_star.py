import pygame
import math
from queue import PriorityQueue

# Initialize the display window
WIDTH = 1000
HEIGHT = 1000
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("A* Path Finding")

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

class Cell:
    def __init__(self, row, col, dimensions, total_rows):
        self.row = row
        self.col = col
        self.x = row * dimensions
        self.y = col * dimensions
        self.color = WHITE
        self.neighbors = []
        self.dimensions = dimensions
        self.total_rows = total_rows


    def get_position(self):
        return self.row, self.col

    
    def is_closed(self):
        return self.color == RED

    
    def is_open(self):
        return self.color == GREEN


    def is_barrier(self):
        return self.color == BLACK

    
    def is_start(self):
        return self.color == ORANGE


    def is_end(self):
        return self.color == TURQUOISE


    def reset(self):
        self.color = WHITE


    def make_closed(self):
        self.color = RED


    def make_open(self):
        self.color = GREEN


    def make_barrier(self):
        self.color = BLACK


    def make_start(self):
        self.color = ORANGE


    def make_end(self):
        self.color = TURQUOISE

    
    def make_path(self):
        self.color = PURPLE


    def draw(self, WINDOW):
        pygame.draw.rect(WINDOW, self.color, (self.x, self.y, self.dimensions, self.dimensions))

    
    def update_neighbors(self, grid):
        self.neighbors = []
        # Down
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row + 1][self.col])

        # Up
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row - 1][self.col])

        # Right
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col + 1])

        # Left
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col - 1])


    def __lt__(self, other):
        return False


def calc_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def draw_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()


def a_star(draw, grid, start, end):
    count = 0
    open = PriorityQueue()
    open.put((0, count, start))
    came_from = {}
    g_score = {cell: float('inf') for row in grid for cell in row}
    g_score[start] = 0
    f_score = {cell: float('inf') for row in grid for cell in row}
    f_score[start] = calc_distance(start.get_position(), end.get_position())
    open_hash = {start} # For checking if cell in open
    
    while open:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            
        current = open.get()[2]
        open_hash.remove(current)
        
        # Draw path
        if current == end:
            draw_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors:
            tmp_g = g_score[current] + 1
            if tmp_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tmp_g
                f_score[neighbor] = tmp_g + calc_distance(neighbor.get_position(), end.get_position())
                if neighbor not in open_hash:
                    count += 1
                    open.put((f_score[neighbor], count, neighbor))
                    open_hash.add(neighbor)
                    neighbor.make_open()
        
        draw()

        if current != start and current != end:
            current.make_closed()

    return False


def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            cell = Cell(i, j, gap, rows)
            grid[i].append(cell)

    return grid


def draw_grid(window, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(window, GREY, (0, i * gap), (width, i * gap))

    for j in range(rows):
        pygame.draw.line(window, GREY, (j * gap, 0), (j * gap, width))


def draw(window, grid, rows, width):
    window.fill(WHITE)
    for row in grid:
        for cell in row:
            cell.draw(window)

    draw_grid(window, rows, width)
    pygame.display.update()


def get_clicked_cell(position, rows, width):
    gap = width // rows
    y, x = position
    row = y // gap
    col = x // gap
    
    return row, col


def main(window, width):
    ROWS = 100
    grid = make_grid(ROWS, width)

    start = None
    end = None
    run = True

    while run:
        draw(window, grid, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

            # Left click
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_cell(pos, ROWS, width)
                cell = grid[row][col]
                if not start and cell != end:
                    start = cell
                    start.make_start()

                elif not end and cell != start:
                    end = cell
                    end.make_end()

                elif cell != start and cell != end:
                    cell.make_barrier()

            # Right click
            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_cell(pos, ROWS, width)
                cell = grid[row][col]
                cell.reset()
                if cell == start:
                    start = None
                elif cell == end:
                    end = None

            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and start and end:
                for row in grid:
                    for cell in row:
                        cell.update_neighbors(grid)
                
                a_star(lambda: draw(window, grid, ROWS, width), grid, start, end)

            if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                start = None
                end = None
                grid = make_grid(ROWS, width)
            

    pygame.quit()






main(WINDOW, WIDTH)