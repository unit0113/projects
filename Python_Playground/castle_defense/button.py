import pygame


class Button():
	def __init__(self, x: int, y: int, image: pygame.surface.Surface) -> None:
		self.image = image
		self.rect = self.image.get_rect()
		self.rect.topleft = (x, y)
		self.clicked = False

	def draw(self, window: pygame.surface.Surface) -> bool:
		action = False
		pos = pygame.mouse.get_pos()

		# Check mouseover and clicked conditions
		if self.rect.collidepoint(pos):
			if pygame.mouse.get_pressed()[0] == 1 and self.clicked == False:
				self.clicked = True
				action = True

		if pygame.mouse.get_pressed()[0] == 0:
			self.clicked = False

		# Draw
		window.blit(self.image, self.rect)

		return action
	