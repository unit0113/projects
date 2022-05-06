import pygame
pygame.init()
import random

# Initialize the display window
screen_height = pygame.display.get_desktop_sizes()[0][1]
WIDTH = HEIGHT = round(screen_height * 0.75)
WINDOW = pygame.display.set_mode((WIDTH + 150, HEIGHT))
pygame.display.set_caption("Soduku")
FONT = pygame.font.SysFont('verdana', 25, bold=False)

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


class Cell:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.x = col * WIDTH // 9 
        self.y = row * HEIGHT // 9
        self.color = WHITE
        self.payload = ' '
        self.dimensions = WIDTH // 9

    def draw(self, window):
        pygame.draw.rect(window, self.color, (self.x, self.y, self.dimensions, self.dimensions))
        cell_text = FONT.render(self.payload, 1, BLACK)
        window.blit(cell_text, (self.x + self.dimensions//2 - cell_text.get_width()//2, self.y + self.dimensions//2 - cell_text.get_height()//2))


class Grid:
    def __init__(self):
        self.grid = []
        for i in range(9):
            self.grid.append([])
            for j in range(9):
                cell = Cell(i, j)
                self.grid[i].append(cell)

        self.generate_solution()
        self.decimate()


    def valid_location(self,row, col, number):
        if self.num_used_in_row(row, number):
            return False

        elif self.num_used_in_column(col, number):
            return False

        elif self.num_used_in_subgrid(row, col, number):
            return False

        return True

    
    def num_used_in_row(self, row, number):
        for i in range(9):
            if self.grid[row][i].payload == str(number):
                return True

        return False


    def num_used_in_column(self, col,number):
        for i in range(9):
            if self.grid[i][col].payload == str(number):
                return True

        return False


    def num_used_in_subgrid(self, row, col, number):
        sub_row = (row // 3) * 3
        sub_col = (col // 3) * 3

        for i in range(sub_row, (sub_row + 3)): 
            for j in range(sub_col, (sub_col + 3)): 
                if self.grid[i][j].payload == str(number):
                    return True

        return False


    def find_empty_square(self):
        for i in range(9):
            for j in range(9):
                if self.grid[i][j].payload == ' ':
                    return (i,j)
        return False


    def generate_solution(self):
        number_list = [num for num in range(1, 10)]
        for cell in range(81):
            row = cell // 9
            col = cell % 9
            if self.grid[row][col].payload == ' ':
                random.shuffle(number_list)
                for num in number_list:
                    if self.valid_location(row, col, num):
                        self.grid[row][col].payload = str(num)
                        if not self.find_empty_square():
                            return True
                        else:
                            if self.generate_solution():
                                #if the grid is full
                                return True
                break
            
        self.grid[row][col].payload = ' '  
        return False


    def get_non_empty_squares(self):
        non_empty_squares = []
        for row in range(len(self.grid)):
            for col in range(len(self.grid)):
                if self.grid[row][col].payload != 0:
                    non_empty_squares.append((row, col))

        random.shuffle(non_empty_squares)

        return non_empty_squares


    def decimate(self):
		#get all non-empty squares from the grid
        non_empty_squares = self.get_non_empty_squares()
        non_empty_squares_count = len(non_empty_squares)

        while non_empty_squares_count > 17:
            row, col = non_empty_squares.pop()
            non_empty_squares_count -= 1
            self.grid[row][col].payload = ' '

        return


    def draw_grid_lines(self, window):
        gap = WIDTH // 9
        # Draw horz lines
        for i in range(10):
            if i % 3 == 0:
                pygame.draw.line(window, GREY, (0, i * gap), (WIDTH, i * gap), width=5)
            else:
                pygame.draw.line(window, GREY, (0, i * gap), (WIDTH, i * gap))

        # Draw vert lines
        for j in range(10):
            if j % 3 == 0:
                pygame.draw.line(window, GREY, (j * gap, 0), (j * gap, WIDTH), width=5)
            else:
                pygame.draw.line(window, GREY, (j * gap, 0), (j * gap, WIDTH))

    
    def draw_grid(self, window):
        # Draw cells
        for row in self.grid:
            for cell in row:
                cell.draw(window)

        self.draw_grid_lines(window)


class Buttons:
    def __init__(self):
        self.buttons = []
        payloads = [' '] + [str(num) for num in range(1,10)]
        y = 60
        for row, payload in enumerate(payloads):
            new_cell = Cell(row, 9)
            new_cell.payload = payload
            new_cell.x = WIDTH + 45
            new_cell.y = y
            y += 90
            new_cell.dimensions = 60
            self.buttons.append(new_cell)


    def draw(self, window):
        for button in self.buttons:
            button.draw(window)

        x1 = WIDTH + 45
        x2 = WIDTH + 105
        # Draw horz lines
        y = 60
        for i in range(20):
            pygame.draw.line(window, GREY, (x1, y), (x2, y), width=3)
            if i % 2 == 0:
                y += 60
            else:
                y += 30

        # Draw vert lines
        y = 60
        for _ in range(10):
            pygame.draw.line(window, GREY, (x1, y), (x1, y + 60), width=3)
            pygame.draw.line(window, GREY, (x2, y), (x2, y + 60), width=3)
            y += 90




def draw(window, grid, buttons):
    window.fill(WHITE)

    grid.draw_grid(window)
    buttons.draw(window)

    pygame.display.update()


def main():
    clock = pygame.time.Clock()
    grid = Grid()
    buttons = Buttons()

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        draw(WINDOW, grid, buttons)

if __name__ == "__main__":
    main()