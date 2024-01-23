import pygame
import random
import math
import os

from src.player import Player
from src.enemy import Enemy
from src.tilemap import Tilemap
from src.utils import load_image, load_images
from src.settings import WIDTH, HEIGHT, FPS
from src.clounds import Clouds
from src.animation import Animation
from src.particle import Particle
from src.spark import Spark


class Game:
    def __init__(self) -> None:
        pygame.init()
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        self.display = pygame.Surface((WIDTH // 2, HEIGHT // 2), pygame.SRCALPHA)
        self.display_2 = pygame.Surface((WIDTH // 2, HEIGHT // 2))
        self.clock = pygame.time.Clock()

        pygame.display.set_caption("Ninja Game")

        self._load_assets()

        self.player = Player(self, (50, 50), (8, 15))
        self.tile_map = Tilemap(self, tile_size=16)
        self.level = 0
        self.load_level(self.level)
        self.clouds = Clouds(self.assets["clouds"], 16)

    def _load_assets(self) -> None:
        self.assets = {}
        self.assets["player"] = load_image("entities/player.png")

        for folder in ["decor", "grass", "large_decor", "stone"]:
            self.assets[folder] = load_images(f"tiles/{folder}")

        self.assets["background"] = load_image("background.png")
        self.assets["clouds"] = load_images("clouds")

        self.assets["player/idle"] = Animation(
            load_images("entities/player/idle"), img_dur=6
        )
        self.assets["player/run"] = Animation(
            load_images("entities/player/run"), img_dur=4
        )
        self.assets["player/jump"] = Animation(
            load_images("entities/player/jump"), img_dur=5
        )
        self.assets["player/slide"] = Animation(
            load_images("entities/player/slide"), img_dur=5
        )
        self.assets["player/wall_slide"] = Animation(
            load_images("entities/player/wall_slide"), img_dur=5
        )

        self.assets["enemy/idle"] = Animation(
            load_images("entities/enemy/idle"), img_dur=6
        )
        self.assets["enemy/run"] = Animation(
            load_images("entities/enemy/run"), img_dur=4
        )
        self.assets["gun"] = load_image("gun.png")
        self.assets["projectile"] = load_image("projectile.png")

        self.assets["particle/leaf"] = Animation(
            load_images("particles/leaf"), img_dur=20, loop=False
        )

        self.assets["particle/particle"] = Animation(
            load_images("particles/particle"), img_dur=6, loop=False
        )

        self.sfx = {}
        for effect, volume in [
            ("jump", 0.7),
            ("dash", 0.3),
            ("hit", 0.8),
            ("shoot", 0.4),
            ("ambience", 0.2),
        ]:
            self.sfx[effect] = pygame.mixer.Sound(f"assets/sfx/{effect}.wav")
            self.sfx[effect].set_volume(volume)

    def load_level(self, map_id):
        self.tile_map.load(f"assets/maps/{map_id}.json")
        # Particles
        self.leaf_spawners = []
        for tree in self.tile_map.extract([("large_decor", 2)], keep=True):
            self.leaf_spawners.append(
                pygame.Rect(4 + tree["pos"][0], 4 + tree["pos"][1], 23, 13)
            )
        self.particles = []
        self.projectiles = []
        self.sparks = []

        # Spawn points
        self.enemies = []
        for spawner in self.tile_map.extract([("spawners", 0), ("spawners", 1)]):
            if spawner["variant"] == 0:
                self.player.pos = spawner["pos"]
            else:
                self.enemies.append(Enemy(self, spawner["pos"], (8, 15)))

        self.movement = [0, 0]
        self.dead = 0
        self.player.air_time = 0
        self.transition = -30

        # Camera vars
        self.scroll = [0, 0]
        self.screen_shake = 0

    def run(self) -> None:
        pygame.mixer.music.load("assets/music.wav")
        pygame.mixer.music.set_volume(0.5)
        pygame.mixer.music.play(-1)
        self.sfx["ambience"].play(-1)

        while True:
            self.clock.tick(FPS)

            self.screen_shake = max(0, self.screen_shake - 1)

            if not len(self.enemies):
                self.transition += 1
                if self.transition > 30:
                    self.level += 1
                    self.load_level(self.level % len(os.listdir("assets/maps")))
            if self.transition < 0:
                self.transition += 1

            if self.dead:
                self.dead += 1
                if self.dead >= 10:
                    self.transition = min(30, self.transition + 1)
                if self.dead > 40:
                    self.load_level(self.level)

            # Update camera
            self.scroll[0] += (
                self.player.rect().centerx - WIDTH / 4 - self.scroll[0]
            ) / 30
            self.scroll[1] += (
                self.player.rect().centery - HEIGHT / 4 - self.scroll[1]
            ) / 30
            render_scroll = (int(self.scroll[0]), int(self.scroll[1]))

            # Event handler
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # Quit if escape is pressed
                        pygame.quit()
                        quit()
                    if event.key == pygame.K_r:
                        self.level = 0
                        self.load_level(self.level)
                    if event.key == pygame.K_LEFT:
                        self.movement[0] = 1
                    if event.key == pygame.K_RIGHT:
                        self.movement[1] = 1
                    if event.key == pygame.K_UP:
                        if self.player.jump():
                            self.sfx["jump"].play()
                    if event.key == pygame.K_x:
                        self.player.dash()
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT:
                        self.movement[0] = 0
                    if event.key == pygame.K_RIGHT:
                        self.movement[1] = 0

            # Update game
            for rect in self.leaf_spawners:
                if random.random() * 49999 < rect.width * rect.height:
                    pos = (
                        rect.x + random.random() * rect.width,
                        rect.y + random.random() * rect.height,
                    )
                    self.particles.append(
                        Particle(
                            self,
                            "leaf",
                            pos,
                            velocity=[-0.1, 0.3],
                            frame=random.randint(0, 20),
                        )
                    )

            for enemy in self.enemies.copy():
                kill = enemy.update(self.tile_map, (0, 0))
                if kill:
                    self.enemies.remove(enemy)

            # [[x, y], direction, timer]
            for projectile in self.projectiles.copy():
                projectile[0][0] += projectile[1]
                projectile[2] += 1
                if self.tile_map.solid_tile_check(projectile[0]):
                    self.projectiles.remove(projectile)
                    for _ in range(4):
                        self.sparks.append(
                            Spark(
                                projectile[0],
                                random.random()
                                - 0.5
                                + (math.pi if projectile[1] > 0 else 0),
                                2 + random.random(),
                            )
                        )
                elif projectile[2] > 360:
                    self.projectiles.remove(projectile)

                elif self.player.dashing < 50:
                    if self.player.rect().collidepoint(projectile[0]):
                        self.sfx["hit"].play()
                        self.projectiles.remove(projectile)
                        self.dead += 1
                        self.screen_shake = max(16, self.screen_shake)
                        for _ in range(30):
                            angle = random.random() * math.pi * 2
                            speed = 5 * random.random()
                            self.sparks.append(
                                Spark(
                                    self.player.rect().center,
                                    angle,
                                    2 + random.random(),
                                )
                            )
                            self.particles.append(
                                Particle(
                                    self,
                                    "particle",
                                    self.player.rect().center,
                                    [
                                        math.cos(angle + math.pi) * speed * 0.5,
                                        math.sin(angle + math.pi) * speed * 0.5,
                                    ],
                                    frame=random.randint(0, 7),
                                )
                            )

            for spark in self.sparks.copy():
                kill = spark.update()
                if kill:
                    self.sparks.remove(spark)

            if not self.dead:
                self.player.update(
                    self.tile_map, (self.movement[1] - self.movement[0], 0)
                )

            self.clouds.update()
            for particle in self.particles.copy():
                kill = particle.update()
                if particle.p_type == "leaf":
                    particle.pos[0] += math.sin(particle.animation.frame * 0.025) * 0.3
                if kill:
                    self.particles.remove(particle)

            # Draw
            self.display.fill((0, 0, 0, 0))
            self.display_2.blit(self.assets["background"], (0, 0))
            self.clouds.draw(self.display_2, offset=render_scroll)
            self.tile_map.draw(self.display, offset=render_scroll)
            for enemy in self.enemies:
                enemy.draw(self.display, offset=render_scroll)

            if not self.dead:
                self.player.draw(self.display, offset=render_scroll)

            projectile_img = self.assets["projectile"]
            for projectile in self.projectiles:
                self.display.blit(
                    projectile_img,
                    (
                        projectile[0][0]
                        - projectile_img.get_width() // 2
                        - render_scroll[0],
                        projectile[0][1]
                        - projectile_img.get_height() // 2
                        - render_scroll[1],
                    ),
                )

            for spark in self.sparks.copy():
                spark.draw(self.display, offset=render_scroll)

            display_mask = pygame.mask.from_surface(self.display)
            display_sillhouette = display_mask.to_surface(
                setcolor=(0, 0, 0, 180), unsetcolor=(0, 0, 0, 0)
            )
            for offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                self.display_2.blit(display_sillhouette, offset)

            for particle in self.particles:
                particle.draw(self.display, offset=render_scroll)

            if self.transition:
                transition_surf = pygame.Surface(self.display.get_size())
                pygame.draw.circle(
                    transition_surf,
                    (255, 255, 255),
                    (self.display.get_width() // 2, self.display.get_height() // 2),
                    (30 - abs(self.transition)) * 8,
                )
                transition_surf.set_colorkey((255, 255, 255))
                self.display.blit(transition_surf, (0, 0))

            self.display_2.blit(self.display, (0, 0))

            screen_shake_offset = (
                random.random() * self.screen_shake - self.screen_shake / 2,
                random.random() * self.screen_shake - self.screen_shake / 2,
            )
            self.window.blit(
                pygame.transform.scale_by(self.display_2, 2), screen_shake_offset
            )
            pygame.display.update()


def main() -> None:
    Game().run()


if __name__ == "__main__":
    main()
