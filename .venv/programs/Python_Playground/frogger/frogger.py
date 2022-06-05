import pygame
import random


# Constants
WIDTH = 1400
HEIGHT = 1000
FPS = 60
JUMP_LENGTH = 20
LANES = 12
LIVES = 3

# Colors
GREEN = (0, 255, 0)
YELLOW = (220, 220, 40)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
L_GRAY = (200, 200, 200)
D_GRAY = (70, 70, 70)


class Frog:
    def __init__(self, window):
        self.window = window
        self.img_frwd = pygame.image.load(r'Python_Playground\frogger\frog.png')
        self.img_back = pygame.image.load(r'Python_Playground\frogger\frog_back.png')
        self.img_left = pygame.image.load(r'Python_Playground\frogger\frog_left.png')
        self.img_right = pygame.image.load(r'Python_Playground\frogger\frog_right.png')
        self.img_death = pygame.image.load(r'Python_Playground\frogger\frog_death.png')
        self.image = self.img_frwd
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect.x = WIDTH // 2 - self.image.get_width() // 2
        self.rect.y = HEIGHT - self.image.get_height() - 10


    def draw(self):
        self.mask = pygame.mask.from_surface(self.image)
        self.window.blit(self.image, (self.rect.x, self.rect.y))

    
    def left(self):
        self.image = self.img_left
        self.rect.move_ip(-JUMP_LENGTH, 0)


    def right(self):
        self.image = self.img_right
        self.rect.move_ip(JUMP_LENGTH, 0)


    def up(self):
        self.image = self.img_frwd
        self.rect.move_ip(0, -JUMP_LENGTH)


    def down(self):
        self.image = self.img_back
        self.rect.move_ip(0, JUMP_LENGTH)

    def reset(self):
        self.rect.x = WIDTH // 2 - self.image.get_width() // 2
        self.rect.y = HEIGHT - self.image.get_height() - 10


class Car:
    def __init__(self, lane_y, x_vel):
        self.img_car_1_left = pygame.image.load(r'Python_Playground\frogger\car_1.png')
        self.img_car_1_right = pygame.transform.flip(self.img_car_1_left.copy(), True, False)
        self.img_car_2_right = pygame.image.load(r'Python_Playground\frogger\car_2.png')
        self.img_car_2_left = pygame.transform.flip(self.img_car_2_right.copy(), True, False)
        self.img_car_3_right = pygame.image.load(r'Python_Playground\frogger\car_3.png')
        self.img_car_3_left = pygame.transform.flip(self.img_car_3_right.copy(), True, False)
        self.img_car_4_left = pygame.image.load(r'Python_Playground\frogger\car_4.png')
        self.img_car_4_right = pygame.transform.flip(self.img_car_4_left.copy(), True, False)
        self.left_images = [self.img_car_1_left, self.img_car_2_left, self.img_car_3_left, self.img_car_4_left]
        self.right_images = [self.img_car_1_right, self.img_car_2_right, self.img_car_3_right, self.img_car_4_right]

        if x_vel < 0:
            self.image = random.choice(self.left_images)
        else:
            self.image = random.choice(self.right_images)

        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect.x = 0 - self.image.get_width() if x_vel > 0 else WIDTH
        self.rect.y = lane_y + random.randint(-10, 10)
        self.x_vel = x_vel


    def update(self):
        self.rect.move_ip(self.x_vel, 0)

    
    def draw(self, window):
        window.blit(self.image, (self.rect.x, self.rect.y))



class Truck:
    def __init__(self, lane_y, x_vel):
        self.img_truck_left = pygame.image.load(r'Python_Playground\frogger\truck.png')
        self.img_truck_right = pygame.transform.flip(self.img_truck_left.copy(), True, False)

        if x_vel < 0:
            self.image = self.img_truck_left
        else:
            self.image = self.img_truck_right

        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect.x = 0 - self.image.get_width() if x_vel > 0 else WIDTH
        self.rect.y = lane_y + random.randint(-10, 10)
        self.x_vel = x_vel

    
    def update(self):
        self.rect.move_ip(self.x_vel, 0)

    
    def draw(self, window):
        window.blit(self.image, (self.rect.x, self.rect.y))


class Game:
    def __init__(self, window):
        self.window = window
        self.level = 0
        self.lives = LIVES
        self.frog = Frog(self.window)
        self.background = pygame.image.load(r'Python_Playground\frogger\background-roadsonly.png')
        self.background = pygame.transform.scale(self.background, (WIDTH, HEIGHT))
        self.lanes_y = [num * HEIGHT // LANES + HEIGHT // (LANES * 2) for num in range(LANES)]
        self.possible_vehicles = ['Car'] * 4 + ['Truck']
        self.start_round()


    def get_random_speed(self):
        return (random.randint(5, 10) + self.level) * random.choice([-1, 1])            


    def start_round(self):
        self.lane_speeds = [0, 0, self.get_random_speed(), self.get_random_speed(), self.get_random_speed(), self.get_random_speed(),
                            0, self.get_random_speed(), self.get_random_speed(), self.get_random_speed(), self.get_random_speed(), 0]
        self.frog.reset()
        self.level += 1
        self.vehciles = []
        self.draw()

    
    def update_vehicles(self):
        # Check death
        if pygame.sprite.spritecollide(self.frog, self.vehciles, False):
            self.death()

        # Move existing
        for vehicle in self.vehciles[:]:
            vehicle.update()

            # Delete if off screen
            if vehicle.rect.x > WIDTH or vehicle.rect.x < 0 - vehicle.image.get_width():
                self.vehciles.remove(vehicle)

        # Create new
        for lane, lane_y in enumerate(self.lanes_y):
            lane_speed = self.lane_speeds[lane]
            if lane_speed == 0 or random.randint(1, 100) > 2:
                continue

            if random.choice(self.possible_vehicles) == 'Car':
                self.vehciles.append(Car(lane_y, lane_speed))
            else:
                self.vehciles.append(Truck(lane_y, lane_speed))


    def update(self):
        if self.frog.rect.y < self.lanes_y[1]:
            pygame.time.wait(500)
            self.start_round()
        
        self.update_vehicles()

        
    def draw_vehicles(self):
        for vehicle in self.vehciles:
            vehicle.draw(self.window)            


    def draw(self):
        self.window.blit(self.background, (0, 0))
        level_text = FONT.render(f'Level: {self.level}', 1, BLACK)
        self.window.blit(level_text, (10, 20 - level_text.get_height() // 2))
        self.frog.draw()
        self.draw_vehicles()
        pygame.display.update()


    def death(self):
        self.frog.image = self.frog.img_death
        self.draw()
        pygame.time.wait(500)
        self.lives -= 1
        if self.lives == 0:
            self.game_over()
        self.level -= 1
        self.start_round()

    
    def game_over(self):
        self.window.fill(BLACK)
        death_text = FONT2.render(f'You Died...', 1, GREEN)
        self.window.blit(death_text, (WIDTH // 2 - death_text.get_width() // 2, HEIGHT // 4 - death_text.get_height() // 2))
        instructions_text = FONT2.render('Press C to play again, or press Q to quit.', 1, GREEN)
        self.window.blit(instructions_text, (WIDTH // 2 - instructions_text.get_width() // 2, HEIGHT // 4 + 25 - instructions_text.get_height() // 2))
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


def initialize_pygame():
    pygame.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Frogger")
    global FONT
    FONT = pygame.font.SysFont('verdana', 20, bold=False)
    global FONT2
    FONT2 = pygame.font.SysFont('verdana', 30, bold=True)

    game = Game(window)

    return game


def main():
    game = initialize_pygame()
    clock = pygame.time.Clock()
    
    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    quit()

                elif event.key == pygame.K_r:
                    main()

                elif event.key == pygame.K_LEFT:
                    game.frog.left()
                
                elif event.key == pygame.K_RIGHT:
                    game.frog.right()

                elif event.key == pygame.K_UP:
                    game.frog.up()

                elif event.key == pygame.K_DOWN:
                    game.frog.down()

        game.update()
        game.draw()


if __name__ == "__main__":
    main()