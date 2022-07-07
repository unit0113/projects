import pygame
import random


# Constants
WIDTH = 1400
HEIGHT = 1000
PADDLE_SPEED = 15
FPS = 60

# Colors
GREEN = (0, 255, 0)
YELLOW = (220, 220, 40)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
L_GRAY = (200, 200, 200)
D_GRAY = (70, 70, 70)


class GameInformation:
    def __init__(self, left_hits, right_hits, left_score, right_score):
        self.left_hits = left_hits
        self.right_hits = right_hits
        self.left_score = left_score
        self.right_score = right_score


class Paddle:
    def __init__(self, side, window):
        self.window = window
        self.length = HEIGHT // 8
        self.width = WIDTH // 75
        self.color = WHITE
        if side == 'left':
            x = self.width
        else:
            x = WIDTH - self.width * 2
        self.rect = pygame.Rect(x, HEIGHT // 2 - self.length // 2, self.width, self.length)

    def draw(self):
        pygame.draw.rect(self.window, self.color, (self.rect.x, self.rect.y, self.width, self.length))


class Ball:
    def __init__(self, window):
        self.window = window
        self.x_velocity = random.randint(10, 15) * random.choice([1, -1])
        self.y_velocity = random.randint(-5, 5)
        self.size = 10
        self.color = WHITE
        self.rect = pygame.Rect(WIDTH // 2, HEIGHT // 2, self.size, self.size)

    def update(self):
        self.rect.x += self.x_velocity
        self.rect.y += self.y_velocity
        if self.rect.y < 0 or self.rect.y > HEIGHT - self.size:
            self.y_velocity *= -1

        if self.rect.x < 0:
            return 'right'
        if self.rect.x > WIDTH:
            return 'left'

        return False

    def draw(self):
        pygame.draw.rect(self.window, self.color, (self.rect.x, self.rect.y, self.size, self.size))

    def check_bounce(self, paddle):
        if pygame.Rect.colliderect(self.rect, paddle.rect):
            self.x_velocity *= -1
            if paddle.rect.x < WIDTH // 2:
                self.rect.x = paddle.rect.x + paddle.width + 1
                self.x_velocity += 1
            else:
                self.rect.x = paddle.rect.x - self.size - 1
                self.x_velocity -= 1

            self.update_y_velocity(paddle)

            return True
        
        return False

    def update_y_velocity(self, paddle):
        y_diff = (self.rect.y + self.size // 2) - (paddle.rect.y + paddle.length // 2)
        self.y_velocity += y_diff // 6


class PongGame:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.score_left = self.score_right = 0
        self.clock = pygame.time.Clock()
        self.window = self.initialize_pygame()
        self.initialize_round()

    def initialize_pygame(self):
        pygame.init()
        window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Pong")
        global FONT
        FONT = pygame.font.SysFont('verdana', 20, bold=True)
        global FONT2
        FONT2 = pygame.font.SysFont('verdana', 30, bold=True)

        return window

    def initialize_round(self):
        self.left_paddle = Paddle('left', self.window)
        self.right_paddle = Paddle('right', self.window)
        self.ball = Ball(self.window)
        self.reward = self.left_hits = self.right_hits = 0
        self.draw()
        pygame.event.clear()

    def draw_scores(self):
        score_left_text = FONT.render(f'Computer: {self.score_left}', 1, GREEN)
        self.window.blit(score_left_text, (10, 20 - score_left_text.get_height() // 2))
        score_right_text = FONT.render(f'AI: {self.score_right}', 1, GREEN)
        self.window.blit(score_right_text, (WIDTH - score_right_text.get_width() - 10, 20 - score_right_text.get_height() // 2))

    def draw(self):
        self.window.fill(BLACK)
        self.left_paddle.draw()
        self.right_paddle.draw()
        self.ball.draw()
        self.draw_scores()
        pygame.display.update()

    def player_up(self):
        if self.right_paddle.rect.y > 0:
            self.right_paddle.rect.y -= PADDLE_SPEED

    def player_down(self):
        if self.right_paddle.rect.y < HEIGHT - self.right_paddle.length:
            self.right_paddle.rect.y += PADDLE_SPEED

    def paddle_up(self, left):
        if left:
            if self.left_paddle.rect.y > 0:
                self.left_paddle.rect.y -= PADDLE_SPEED
            else:
                return False
        else:
            if self.right_paddle.rect.y > 0:
                self.right_paddle.rect.y -= PADDLE_SPEED
            else:
                return False
        
        return True

    def paddle_down(self, left):
        if left:
            if self.left_paddle.rect.y < HEIGHT - self.left_paddle.length:
                self.left_paddle.rect.y += PADDLE_SPEED
            else:
                return False
        else:
            if self.right_paddle.rect.y < HEIGHT - self.right_paddle.length:
                self.right_paddle.rect.y += PADDLE_SPEED
            else:
                return False
        
        return True

    def computer_move(self):
        if self.ball.rect.y > self.left_paddle.rect.y + self.left_paddle.length // 2 + PADDLE_SPEED * 2 and self.left_paddle.rect.y < HEIGHT - self.left_paddle.length:
            self.left_paddle.rect.y += PADDLE_SPEED
        elif self.ball.rect.y < self.left_paddle.rect.y + self.left_paddle.length // 2 - PADDLE_SPEED * 2 and self.left_paddle.rect.y > 0:
            self.left_paddle.rect.y -= PADDLE_SPEED

    def move_paddle(self, left=True, up=True):
        """
        Move the left or right paddle.
        :returns: boolean indicating if paddle movement is valid. 
                  Movement is invalid if it causes paddle to go 
                  off the screen
        """
        if left:
            if up and self.left_paddle.y - Paddle.VEL < 0:
                return False
            if not up and self.left_paddle.y + Paddle.HEIGHT > self.window_height:
                return False
            self.left_paddle.move(up)
        else:
            if up and self.right_paddle.y - Paddle.VEL < 0:
                return False
            if not up and self.right_paddle.y + Paddle.HEIGHT > self.window_height:
                return False
            self.right_paddle.move(up)

        return True

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.computer_move()
        score = self.ball.update()
        if not score:
            if self.ball.check_bounce(self.left_paddle):
                self.left_hits += 1
            elif self.ball.check_bounce(self.right_paddle):
                self.right_hits += 1

        elif score == 'right':
            self.score_right += 1
        elif score == 'left':
            self.score_left += 1
        
        return GameInformation(self.left_hits, self.right_hits, self.score_left, self.score_right)

    def play_step(self, action):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        if action[0]:
            self.ai_up()
        elif action[2]:
            self.ai_down()

        self.computer_move()
        
        # 3. check if game over
        game_over = self.ball.update()
        if not game_over:
            self.ball.check_bounce(self.left_paddle)

            if self.ball.check_bounce(self.right_paddle):
                self.reward += abs(self.ball.y_velocity)
        
        else:
            if game_over == 'right':
                self.score_right += 1
                game_over = True
                self.reward += 1000
            elif game_over == 'left':
                self.score_left += 1
                game_over = True
                self.reward -= 100

            return self.reward, game_over
        
        # 4. update ui and clock
        self.draw()
        self.clock.tick(FPS)

        # 5. return game over and score
        return self.reward, game_over
