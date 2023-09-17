import pygame

from settings import CASTLE_X

FRAME_LENGTH = 0.05

class Enemy(pygame.sprite.Sprite):
    def __init__(self, hp: int, sprites: dict[str, list[pygame.surface.Surface]], x: int, y: int, speed: int, damage: int) -> None:
        pygame.sprite.Sprite.__init__(self)
        self.alive = True
        self.speed = speed
        self.hp = hp
        self.max_hp = hp
        self.damage = damage
        self.sprites = sprites

        self.frame_index = 0
        self.action = 'walk'
        self.timer = 0

        self.image = self.sprites[self.action][0]
        self.rect = pygame.Rect(0,0, 25, 40)
        self.rect.center = (x, y)

    def update(self, dt: float, window: pygame.surface.Surface) -> None:
        self.timer += dt
        if self.action == 'walk' and self.rect.right < CASTLE_X:
            if self.rect.right < 0:
                self.rect.x += 1
            else:
                self.rect.x += self.speed
        elif self.action == 'walk':
            self._update_action('attack')
        
        # Update sprite animation
        if self.timer > FRAME_LENGTH:
            self.frame_index += 1
            self.timer = 0
            if self.frame_index >= len(self.sprites[self.action]):
                self.frame_index = 0
                if self.action == 'death':
                    self.kill()
                    return

        self.display(window)

    def display(self, window: pygame.surface.Surface) -> None:
        # Draw enemy to screen
        window.blit(self.sprites[self.action][self.frame_index], (self.rect.x - 10, self.rect.y - 15))

    def _update_action(self, action: str) -> None:
        if self.action == action:
            return
        
        self.action = action
        self.frame_index = 0

    def take_hit(self, damage: int) -> int:
        if self.action == 'death':
            return 0
        
        self.hp -= damage
        if self.hp <= 0:
            self._update_action('death')
            return self.max_hp
        return 0
    
    def do_damage(self) -> int:
        if self.action == 'attack' and self.frame_index == len(self.sprites[self.action]) - 1 and self.timer == 0:
            return self.damage
    
        else:
            return 0