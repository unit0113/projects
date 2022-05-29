import pygame
pygame.init()
import random
import time


# Initialize the main window
CELL_DIMENSION = 100
COLUMNS, ROWS = 5, 6
WIDTH = COLUMNS * CELL_DIMENSION
HEIGHT = ROWS * CELL_DIMENSION
HORZ_MARGIN = 200
VERT_MARGIN_TOP = 50
VERT_MARGIN_BOTTOM = 400
TOTAL_WIDTH = WIDTH+2*HORZ_MARGIN
WINDOW = pygame.display.set_mode((TOTAL_WIDTH, HEIGHT+VERT_MARGIN_TOP+VERT_MARGIN_BOTTOM))
pygame.display.set_caption("Wordle")
FONT = pygame.font.SysFont('verdana', 30, bold=False)

# Colors
GREEN = (0, 255, 0)
YELLOW = (220, 220, 40)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
L_GRAY = (200, 200, 200)
D_GRAY = (70, 70, 70)


class Cell:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.dimensions = CELL_DIMENSION
        self.x = col * self.dimensions + HORZ_MARGIN
        self.y = row * self.dimensions + VERT_MARGIN_TOP
        self.color = GRAY
        self.payload = ' '
        self.rect = pygame.Rect(self.x, self.y, self.dimensions, self.dimensions)
        self.font = FONT

    def draw(self, window):
        pygame.draw.rect(window, self.color, (self.x, self.y, self.dimensions, self.dimensions))
        cell_text = self.font.render(self.payload, 1, WHITE)
        window.blit(cell_text, (self.x + self.dimensions//2 - cell_text.get_width()//2, self.y + self.dimensions//2 - cell_text.get_height()//2))


class Grid:
    def __init__(self, window):
        self.get_word_list()
        self.word = random.choice(self.answer_words).upper()
        self.grid = []
        self.active_row = 0
        self.active_column = 0
        self.window = window
        for i in range(ROWS):
            self.grid.append([])
            for j in range(COLUMNS):
                cell = Cell(i, j)
                self.grid[i].append(cell)
        
        self.draw_grid()


    def get_word_list(self):
        # Get possible guesses
        with open(r'Python_Playground\wordle\word_list.txt', 'r') as file:
            self.guess_words = [word.strip() for word in file.readlines()]
        
        # Get possible answers
        with open(r'Python_Playground\wordle\answer_list.txt', 'r') as file:
            self.answer_words = [word.strip() for word in file.readlines()]

    
    def draw_grid_lines(self):
        gap = self.grid[0][0].dimensions
        # Draw horz lines
        for i in range(ROWS+1):
            pygame.draw.line(self.window, WHITE, (HORZ_MARGIN, i * gap + VERT_MARGIN_TOP), (WIDTH + HORZ_MARGIN, i * gap + VERT_MARGIN_TOP), width=5)

        # Draw vert lines
        for j in range(COLUMNS+1):
                pygame.draw.line(self.window, WHITE, (j * gap + HORZ_MARGIN, VERT_MARGIN_TOP), (j * gap + HORZ_MARGIN, HEIGHT + VERT_MARGIN_TOP), width=5)

    
    def draw_grid(self):
        self.window.fill(BLACK)
        # Draw cells
        for row in self.grid:
            for cell in row:
                cell.draw(self.window)

        self.draw_grid_lines()
        pygame.display.update()


    def update(self, event_key, keyboard):
        # Check if row already full
        if self.active_column >= COLUMNS:
            return

        self.grid[self.active_row][self.active_column].payload = chr(event_key).upper()
        self.active_column += 1
        self.draw_grid()
        keyboard.draw_keys()


    def play(self, keyboard):
        # Check if row is full
        if self.active_column != COLUMNS:
            return

        # Check if valid guess
        guess = ''
        for cell in self.grid[self.active_row]:
            guess += cell.payload
        if guess.lower() not in self.guess_words:
            error_text = FONT.render(f'{guess} is not a valid word!', 1, GREEN)
            self.window.blit(error_text, (TOTAL_WIDTH // 2 - error_text.get_width() // 2, VERT_MARGIN_TOP // 2 - error_text.get_height() // 2))
            pygame.display.update()
            return

        word_copy = list(self.word)
        num_correct = 0
        # Check for greens
        for index, cell in enumerate(self.grid[self.active_row]):
            if cell.payload == self.word[index]:
                cell.color = GREEN
                keyboard.color_key_green(cell.payload)
                word_copy[index] = '_'
                num_correct += 1

        # Check for remaining yellows
        for cell in self.grid[self.active_row]:
            if cell.payload in word_copy and cell.color != GREEN:
                cell.color = YELLOW
                keyboard.color_key_yellow(cell.payload)
                word_copy.remove(cell.payload)
            elif cell.color != GREEN:
                keyboard.color_key_gray(cell.payload)

        self.draw_grid()
        keyboard.draw_keys()

        if num_correct == COLUMNS:
            self.win()

        self.active_row += 1
        self.active_column = 0

        if self.active_row == ROWS:
            self.lose()

    
    def delete(self, keyboard):
        # Check if no entries
        if self.active_column == 0:
            return

        self.active_column -= 1
        self.grid[self.active_row][self.active_column].payload = ' '

        self.draw_grid()
        keyboard.draw_keys()


    def win(self):
        time.sleep(1)
        self.window.fill(GRAY)
        greeting_text = FONT.render('Congratulations!!!!', 1, GREEN)
        self.window.blit(greeting_text, (TOTAL_WIDTH // 2 - greeting_text.get_width() // 2, HEIGHT // 2 - greeting_text.get_height() // 2))
        instructions_text_1 = FONT.render('Press C to play again', 1, GREEN)
        self.window.blit(instructions_text_1, (TOTAL_WIDTH // 2 - instructions_text_1.get_width() // 2, HEIGHT // 2 + 30 - instructions_text_1.get_height() // 2))
        instructions_text_2 = FONT.render('Or press Q to quit.', 1, GREEN)
        self.window.blit(instructions_text_2, (TOTAL_WIDTH // 2 - instructions_text_2.get_width() // 2, HEIGHT // 2 + 60 - instructions_text_2.get_height() // 2))
        pygame.display.update()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

                elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    main()

                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    pygame.quit()
                    quit()


    def lose(self):
        time.sleep(1)
        self.window.fill(GRAY)
        greeting_text1 = FONT.render('You have run out of guesses!!', 1, GREEN)
        self.window.blit(greeting_text1, (TOTAL_WIDTH // 2 - greeting_text1.get_width() // 2, HEIGHT // 2 - 30 - greeting_text1.get_height() // 2))
        greeting_text2 = FONT.render(f'The word was {self.word}!', 1, GREEN)
        self.window.blit(greeting_text2, (TOTAL_WIDTH // 2 - greeting_text2.get_width() // 2, HEIGHT // 2 - greeting_text2.get_height() // 2))
        greeting_text3 = FONT.render('Better luck next time!', 1, GREEN)
        self.window.blit(greeting_text3, (TOTAL_WIDTH // 2 - greeting_text3.get_width() // 2, HEIGHT // 2 + 30 - greeting_text3.get_height() // 2))
        instructions_text_1 = FONT.render('Press C to play again', 1, GREEN)
        self.window.blit(instructions_text_1, (TOTAL_WIDTH // 2 - instructions_text_1.get_width() // 2, HEIGHT // 2 + 60 - instructions_text_1.get_height() // 2))
        instructions_text_2 = FONT.render('Or press Q to quit.', 1, GREEN)
        self.window.blit(instructions_text_2, (TOTAL_WIDTH // 2 - instructions_text_2.get_width() // 2, HEIGHT // 2 + 90 - instructions_text_2.get_height() // 2))
        pygame.display.update()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

                elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    main()

                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    pygame.quit()
                    quit()


class Key:
    def __init__(self, row, col, offset):
        self.row = row
        self.col = col
        self.dimensions = int(CELL_DIMENSION * 3 / 4)
        self.x = col * self.dimensions + HORZ_MARGIN // 2 - self.dimensions // 4 + offset
        self.y = row * self.dimensions + VERT_MARGIN_TOP + HEIGHT + VERT_MARGIN_BOTTOM // 4
        self.color = L_GRAY
        self.payload = ' '
        self.rect = pygame.Rect(self.x, self.y, self.dimensions, self.dimensions)
        self.font = FONT

    def draw(self, window):
        pygame.draw.rect(window, self.color, (self.x, self.y, self.dimensions, self.dimensions))
        cell_text = self.font.render(self.payload, 1, WHITE)
        window.blit(cell_text, (self.x + self.dimensions//2 - cell_text.get_width()//2, self.y + self.dimensions//2 - cell_text.get_height()//2))


class BigKey:
    def __init__(self, row, col, offset, window):
        self.row = row
        self.col = col
        self.dimensions = int(CELL_DIMENSION * 3 / 4)
        self.x = col * self.dimensions + HORZ_MARGIN // 2 - self.dimensions // 2 + offset
        self.y = row * self.dimensions + VERT_MARGIN_TOP + HEIGHT + VERT_MARGIN_BOTTOM // 4
        self.color = L_GRAY
        self.payload = ' '
        self.rect = pygame.Rect(self.x, self.y, self.dimensions, self.dimensions * 2)
        self.font = FONT
        self.window = window

    def draw(self, window):
        pygame.draw.rect(window, self.color, (self.x, self.y, self.dimensions * 2, self.dimensions))
        cell_text = self.font.render(self.payload, 1, WHITE)
        window.blit(cell_text, (self.x + self.dimensions - cell_text.get_width()//2, self.y + self.dimensions//2 - cell_text.get_height()//2))
        pygame.draw.line(self.window, WHITE, (self.x, self.y), (self.x, self.y + self.dimensions), width=5)
        pygame.draw.line(self.window, WHITE, (self.x + self.dimensions * 2, self.y), (self.x + self.dimensions * 2, self.y + self.dimensions), width=5)
        pygame.draw.line(self.window, WHITE, (self.x, self.y), (self.x  + self.dimensions * 2, self.y), width=5)
        pygame.draw.line(self.window, WHITE, (self.x, self.y + self.dimensions), (self.x  + self.dimensions * 2, self.y + self.dimensions), width=5)


class Keyboard:
    def __init__(self, window):
        row1 = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P']
        row2 = ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L']
        row3 = ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
        self.letters = [row1, row2, row3]
        self.keys = []
        self.window = window
        offset = 0
        for row_index, row in enumerate(self.letters):
            self.keys.append([])
            for col_index, letter in enumerate(row):
                key = Key(row_index, col_index, offset)
                key.payload = letter
                self.keys[row_index].append(key)
            offset += 35

        # Create backspace
        key = BigKey(row_index, col_index+1, offset-15, self.window)
        key.payload = 'Delete'
        self.keys[row_index].append(key)
        
        self.draw_keys()


    def draw_keys(self):
        # Draw leys
        for row in self.keys:
            for key in row:
                key.draw(self.window)

        self.draw_key_lines()
        pygame.display.update()


    def draw_key_lines(self):
        gap = self.keys[0][0].dimensions

        for row in self.keys:
            # Draw horz lines
            pygame.draw.line(self.window, WHITE, (row[0].x, row[0].y), (row[-1].x + gap, row[0].y), width=5)
            pygame.draw.line(self.window, WHITE, (row[0].x, row[0].y + gap), (row[-1].x + gap, row[0].y + gap), width=5)

            # Draw left vert lines of keys
            for key in row:
                pygame.draw.line(self.window, WHITE, (key.x, key.y), (key.x, key.y + gap), width=5)

            # Draw far right vert line
            if key.payload != 'Delete': 
                pygame.draw.line(self.window, WHITE, (key.x + gap, key.y), (key.x + gap, key.y + gap), width=5)


    def _find_key_index(self, letter):
        for row_index, row in enumerate(self.letters):
            if letter in row:
                col_index = row.index(letter)
                break

        return row_index, col_index


    def color_key_green(self, letter):
        row_index, col_index = self._find_key_index(letter)
        self.keys[row_index][col_index].color = GREEN


    def color_key_yellow(self, letter):
        row_index, col_index = self._find_key_index(letter)
        self.keys[row_index][col_index].color = YELLOW


    def color_key_gray(self, letter):
        row_index, col_index = self._find_key_index(letter)
        self.keys[row_index][col_index].color = D_GRAY


    def check_button_press(self, pos, grid):
        for row in self.keys:
            for key in row:
                if key.rect.collidepoint(pos):
                    if key.payload == 'Delete':
                        grid.delete(self)
                    else:
                        grid.update(ord(key.payload.lower()), self)


def letter_pressed(event_key):
    return event_key >= 97 and event_key <= 122


def main():
    grid = Grid(WINDOW)
    keyboard = Keyboard(WINDOW)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    grid.play(keyboard)
                elif letter_pressed(event.key):
                    grid.update(event.key, keyboard)
                elif event.key == pygame.K_BACKSPACE:
                    grid.delete(keyboard)
            elif pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                keyboard.check_button_press(pos, grid)
        

if __name__ == "__main__":
    main()