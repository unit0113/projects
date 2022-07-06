import pygame
import os
import asset_manager
import store
from settings import WIDTH, HEIGHT, FPS, PLAYER_LIVES, LEVEL_DURATION, BONUS_NO_DAMAGE_TAKEN, BONUS_ALL_ENEMIES_KILLED, WHITE


"""TODO:
power-ups
different ship types
different enemy types, use a factory
mouse controls
fix alignment on scorecard
missiles
"""


class SpaceGame:
    def __init__(self, window):
        self.window = window
        self.background = pygame.transform.scale(pygame.image.load(os.path.join(r'Python_Playground\space_invaders\Assets', 'space.png')), (WIDTH, HEIGHT))
        self.player_lives = PLAYER_LIVES
        self.labels_font = pygame.font.SysFont('verdana', 30, bold=True)
        self.scorecard_font = pygame.font.SysFont('verdana', 25, bold=True)
        self.store_title_font = pygame.font.SysFont('verdana', 40, bold=True)
        self.score = 0
        self.credits = 0
        self.max_level_duration = LEVEL_DURATION * FPS
        self.asset_manager = asset_manager.AssetManager(self.window)
        self.store = store.Store(self.window, self.background, self.asset_manager.player)

        self.initialize_round()

    @property
    def is_round_end(self):
        return self.level_duration > self.max_level_duration and not self.asset_manager.bad_guys

    def initialize_round(self):
        self.asset_manager.new_round()
        self.level_duration = 0
        self.score_start_of_round = self.score
        self.asset_manager.draw()

    def end_round(self):
        self.round_scoreboard()
        self.credits, self.player_lives = self.store.open_store(self.credits, self.player_lives)
        self.initialize_round()

    def round_scoreboard(self):
        self.window.blit(self.background, (0, 0))
        self.asset_manager.draw()

        spacer = 0
        enemies_text = self.scorecard_font.render(f'{"Enemy ships destroyed":.<30}{self.asset_manager.num_baddies_killed_round:.>10}', 1, WHITE)
        self.window.blit(enemies_text, (WIDTH // 2 - enemies_text.get_width() // 2, HEIGHT // 3 - enemies_text.get_height() // 2))

        spacer += 1
        round_points = self.score - self.score_start_of_round
        points_text = self.scorecard_font.render(f'{"Points earned:":.<30}{round_points:.>10}', 1, WHITE)
        self.window.blit(points_text, (WIDTH // 2 - points_text.get_width() // 2, HEIGHT // 3 + 30 * spacer - points_text.get_height() // 2))

        spacer += 1
        level_bonus_pts = int(round_points * (self.asset_manager.level / 10))
        level_bonus_text = self.scorecard_font.render(f'{"Level bonus:":.<30}{level_bonus_pts:.>10}', 1, WHITE)
        self.window.blit(level_bonus_text, (WIDTH // 2 - level_bonus_text.get_width() // 2, HEIGHT // 3 + 30 * spacer - level_bonus_text.get_height() // 2))

        no_damamge_pts = all_baddies_killed_pts = 0
        if self.asset_manager.no_damage_taken:
            spacer += 1
            no_damamge_pts = int(round_points * BONUS_NO_DAMAGE_TAKEN)
            no_damage_text = self.scorecard_font.render(f'{"No damage taken bonus:":.<30}{no_damamge_pts:.>10}', 1, WHITE)
            self.window.blit(no_damage_text, (WIDTH // 2 - no_damage_text.get_width() // 2, HEIGHT // 3 + 30 * spacer - no_damage_text.get_height() // 2))

        if self.asset_manager.no_baddies_escaped:
            spacer += 1
            all_baddies_killed_pts = int(round_points * BONUS_ALL_ENEMIES_KILLED)
            all_killed_text = self.scorecard_font.render(f'{"Total destruction bonus:":.<30}{all_baddies_killed_pts:.>10}', 1, WHITE)
            self.window.blit(all_killed_text, (WIDTH // 2 - all_killed_text.get_width() // 2, HEIGHT // 3 + 30 * spacer - all_killed_text.get_height() // 2))

        spacer += 2
        self.score = self.score_start_of_round + round_points + level_bonus_pts + no_damamge_pts + all_baddies_killed_pts
        total_pts_text = self.scorecard_font.render(f'{"Total score:":.<30}{self.score:.>10}', 1, WHITE)
        self.window.blit(total_pts_text, (WIDTH // 2 - total_pts_text.get_width() // 2, HEIGHT // 3 + 30 * spacer - total_pts_text.get_height() // 2))

        spacer += 3
        instructions_text = self.labels_font.render('Press C to continue.', 1, WHITE)
        self.window.blit(instructions_text, (WIDTH // 2 - instructions_text.get_width() // 2, HEIGHT // 3 + 30 * spacer - instructions_text.get_height() // 2))

        pygame.display.update()

        self.credits += self.score - self.score_start_of_round

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

                elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    return     

    def player_up(self):
        self.asset_manager.player.player_up()

    def player_down(self):
        self.asset_manager.player.player_down()

    def player_left(self):
        self.asset_manager.player.player_left()

    def player_right(self):
        self.asset_manager.player.player_right()

    def player_fire(self):
        self.asset_manager.player_fire()

    def update(self):
        self.level_duration += 1
        if self.level_duration > self.max_level_duration:
            self.asset_manager.new_baddies_generation = False
        self.window.blit(self.background, (0, 0))
        self.asset_manager.update()
        self.asset_manager.draw()
        self.score += self.asset_manager.check_hits()

        if self.is_round_end:
            self.end_round()
        self.check_player_death()

        self._draw_labels()

        pygame.display.update()

    def _draw_labels(self):
        lives_label = self.labels_font.render(f'Lives: {self.player_lives}', 1, WHITE)
        self.window.blit(lives_label, (15, 15))
        level_label = self.labels_font.render(f'Level: {self.asset_manager.level}', 1, WHITE)
        self.window.blit(level_label, (WIDTH - level_label.get_width() - 15, 15))
        score_label = self.labels_font.render(f'Score: {self.score}', 1, WHITE)
        self.window.blit(score_label, (WIDTH // 2 - score_label.get_width() // 2, 15))

    def check_player_death(self):
        if self.asset_manager.player.is_dead:
            if self.player_lives > 0:
                self.player_lives -= 1
                self.asset_manager.player.post_death_actions()
            else:
                self.game_over()
    
    def game_over(self):
        self.window.blit(self.background, (0, 0))
        self.game_over_font = pygame.font.SysFont('verdana', 50, bold=True)

        death_text = self.game_over_font.render(f'You Died...', 1, WHITE)
        self.window.blit(death_text, (WIDTH // 2 - death_text.get_width() // 2, HEIGHT // 4 - death_text.get_height() // 2))

        level_text = self.game_over_font.render(f'You reached level {self.level}', 1, WHITE)
        self.window.blit(level_text, (WIDTH // 2 - level_text.get_width() // 2, HEIGHT // 4 + 50 - level_text.get_height() // 2))

        score_text = self.game_over_font.render(f'Your final score is {self.score}', 1, WHITE)
        self.window.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, HEIGHT // 4 + 100 - score_text.get_height() // 2))

        instructions_text = self.game_over_font.render('Press C to play again, or press Q to quit.', 1, WHITE)
        self.window.blit(instructions_text, (WIDTH // 2 - instructions_text.get_width() // 2, HEIGHT // 4 + 150 - instructions_text.get_height() // 2))
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
    pygame.display.set_caption("Space Invaders")
    pygame.font.init()

    return window


def main():
    window = initialize_pygame()
    space_game = SpaceGame(window)
    clock = pygame.time.Clock()

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        keys = pygame.key.get_pressed()

        if keys[pygame.K_q]:
            pygame.quit()
            quit()

        if keys[pygame.K_r]:
            main()

        if keys[pygame.K_UP]:
            space_game.player_up()

        if keys[pygame.K_DOWN]:
            space_game.player_down()

        if keys[pygame.K_LEFT]:
            space_game.player_left()

        if keys[pygame.K_RIGHT]:
            space_game.player_right()
        
        if keys[pygame.K_SPACE]:
            space_game.player_fire()

        space_game.update()


if __name__ == "__main__":
    main()
