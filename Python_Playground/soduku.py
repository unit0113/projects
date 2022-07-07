import pygame
pygame.init()
import random

# Initialize the main window
screen_height = pygame.display.get_desktop_sizes()[0][1]
WIDTH = HEIGHT = round(screen_height * 0.75)
WINDOW = pygame.display.set_mode((WIDTH + 150, HEIGHT))
pygame.display.set_caption("Soduku")
FONT = pygame.font.SysFont('verdana', 30, bold=False)
LOCKED_FONT = pygame.font.SysFont('verdana', 30, bold=True)


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
        self.lock = False
        self.rect = pygame.Rect(self.x, self.y, self.dimensions, self.dimensions)
        self.font = FONT

    def draw(self, window):
        pygame.draw.rect(window, self.color, (self.x, self.y, self.dimensions, self.dimensions))
        cell_text = self.font.render(self.payload, 1, BLACK)
        window.blit(cell_text, (self.x + self.dimensions//2 - cell_text.get_width()//2, self.y + self.dimensions//2 - cell_text.get_height()//2))


class Grid:
    def __init__(self):
        self.valid_numbers = {str(num) for num in range(1,10)}
        self.grid = []
        for i in range(9):
            self.grid.append([])
            for j in range(9):
                cell = Cell(i, j)
                self.grid[i].append(cell)

        self.generate_solution()
        self.decimate()


    def valid_location(self, row, col, number):
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


    def num_used_in_column(self, col, number):
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
                if self.grid[row][col].payload != ' ':
                    non_empty_squares.append((row, col))

        random.shuffle(non_empty_squares)

        return non_empty_squares


    def decimate(self):
		# Get all non-empty squares from the grid
        non_empty_squares = self.get_non_empty_squares()
        non_empty_squares_count = len(non_empty_squares)

        while non_empty_squares_count > 17:
            row, col = non_empty_squares.pop()
            non_empty_squares_count -= 1
            self.grid[row][col].payload = ' '

        # Lock contents of remaining cells
        non_empty_squares = self.get_non_empty_squares()
        for row, col in non_empty_squares:
            cell = self.grid[row][col]
            cell.lock = True
            cell.font = LOCKED_FONT

        return


    def check_win(self):
        if (self.check_rows_win() and
            self.check_cols_win() and
            self.check_subgrids_win()):
            return True

        return False


    def check_rows_win(self):
        for row in range(9):
            row_set = {cell.payload for cell in self.grid[row]}
            if row_set != self.valid_numbers:
                return False

        return True


    def check_cols_win(self):
        for col in range(9):
            col_set = set()
            for row in range(9):
                col_set.add(self.grid[row][col].payload)
            if col_set != self.valid_numbers:
                return False

        return True


    def check_subgrids_win(self):
        # Go to subgrid
        for sub_row in range(3):
            for sub_col in range(3):

                # Check subgrid
                grid_set = set()
                for row in range(sub_row * 3, (sub_row * 3 + 3)): 
                    for col in range(sub_col * 3, (sub_col * 3 + 3)): 
                        grid_set.add(self.grid[row][col].payload)
                if grid_set != self.valid_numbers:
                    return False

        return True


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


    def check_button_press(self, pos, active):
        for row in range(9):
            for col in range(9):
                cell = self.grid[row][col]
                if cell.rect.collidepoint(pos) and not cell.lock:
                    cell.payload = active
                    if self.check_win():
                        return True
                    return False


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
            new_cell.rect = pygame.Rect(new_cell.x, new_cell.y, new_cell.dimensions, new_cell.dimensions)
            self.buttons.append(new_cell)

        self.active = self.buttons[0]
        self.active.color = GREEN


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

        
    def check_button_press(self, pos):
        for button in self.buttons:
            if button.rect.collidepoint(pos):
                self.active.color = WHITE
                self.active = button
                self.active.color = GREEN


def draw(window, grid, buttons):
    window.fill(WHITE)

    grid.draw_grid(window)
    buttons.draw(window)

    pygame.display.update()


def initilize_game():
    grid = Grid()
    buttons = Buttons()
    return grid, buttons


def endgame(window):
    window.fill(WHITE)
    clock = pygame.time.Clock()
    greeting_text = FONT.render('Congratulations!!!!', 1, PURPLE)
    window.blit(greeting_text, (WIDTH // 2 + 75- greeting_text.get_width() // 2, HEIGHT // 2 - greeting_text.get_height() // 2))
    instructions_text = FONT.render('Press C to play again, or press Q to quit.', 1, PURPLE)
    window.blit(instructions_text, (WIDTH // 2 + 75 - instructions_text.get_width() // 2, HEIGHT // 2 + 25 - instructions_text.get_height() // 2))
    pygame.display.update()

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                main()

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                pygame.quit()
                quit()


def main():
    clock = pygame.time.Clock()
    grid, buttons = initilize_game()

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                buttons.check_button_press(pos)
                if grid.check_button_press(pos, buttons.active.payload):
                    endgame(WINDOW)

        draw(WINDOW, grid, buttons)
        

if __name__ == "__main__":
    main()