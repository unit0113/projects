import pygame
import pymunk
import pymunk.pygame_util
import math
pygame.init()


WIDTH, HEIGHT = 1200, 1200
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Physics Simulation")


def calculate_distance(p1, p2):
    return math.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)


def calculate_angle(p1, p2):
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


def draw(space, window, draw_options):
    window.fill('black')
    space.debug_draw(draw_options)
    pygame.display.update()


def create_boundaries(space, window):
    width, height = window.get_size()
    rects = [
        [(width / 2, height - 10), (width, 20)],
        [(width / 2, 10), (width, 20)],
        [(10, height / 2), (20, height)],
        [(width - 10, height / 2), (20, height)]
    ]

    for pos, size in rects:
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = pos
        shape = pymunk.Poly.create_box(body, size)
        shape.elasticity = 0.4
        shape.friction = 0.5
        space.add(body, shape)


def create_ball(space, radius, mass, pos):
    body = pymunk.Body()
    body.position = pos
    shape = pymunk.Circle(body, radius)
    shape.elasticity = 0.9
    shape.friction = 0.4
    shape.mass = mass
    shape.color = (255, 0, 0, 100)
    space.add(body, shape)
    return shape


def create_structures(space, window):
    width, height = window.get_size()
    BROWN = (139, 69, 19, 100)
    rects = [
        [(600, height - 120), (40, 200), BROWN, 100],
        [(900, height - 120), (40, 200), BROWN, 100],
        [(750, height - 240), (340, 40), BROWN, 150],
    ]

    for pos, size, color, mass in rects:
        body = pymunk.Body()
        body.position = pos
        shape = pymunk.Poly.create_box(body, size, radius=2)
        shape.color = color
        shape.mass = mass
        shape.elasticity = 0.4
        shape.friction = 0.4
        space.add(body, shape)


def create_pendulum(space):
    rot_center_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    rot_center_body.position = (300, 300)

    body = pymunk.Body()
    body.position = (300, 300)
    line = pymunk.Segment(body, (0, 0), (255, 0), 5)
    circle = pymunk.Circle(body, 40, (255, 0))
    line.friction = circle.friction = 1
    line.mass = 8
    circle.mass = 30
    circle.elasticity = .95

    rot_center_joint = pymunk.PinJoint(body, rot_center_body, (0, 0), (0, 0))
    space.add(circle, line, body, rot_center_joint)


def main(window):
    clock = pygame.time.Clock()
    FPS = 60
    delta_time = 1 / FPS

    space = pymunk.Space()
    space.gravity = (0, 981)

    create_boundaries(space, window)
    create_structures(space, window)
    create_pendulum(space)

    draw_options = pymunk.pygame_util.DrawOptions(window)

    pressed_pos = None
    ball = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            elif pygame.mouse.get_pressed()[0]:
                pressed_pos = pygame.mouse.get_pos()
                if not ball:
                    ball = create_ball(space, 30, 10, pressed_pos)
                else:
                    angle = calculate_angle(ball._get_body().position, pressed_pos) - ball._get_body().angle
                    force = calculate_distance(ball._get_body().position, pressed_pos) * 25
                    fx = force * math.cos(angle)
                    fy = force * math.sin(angle)
                    ball.body.apply_impulse_at_local_point((fx, fy), (0, 0))

            elif pygame.mouse.get_pressed()[2] and ball:
                space.remove(ball, ball.body)
                ball = None
                pressed_pos = None

        draw(space, window, draw_options)
        space.step(delta_time)
        clock.tick(60)


if __name__ == "__main__":
    main(WINDOW)