import pygame
from pygame.math import Vector2 as Vector
from pygame.gfxdraw import bezier


class Node:
    def __init__(self, pos: tuple[float, float], size):
        self.pos = Vector(pos)
        self.size = size

    def calc_front(self, diff: Vector) -> tuple[Vector, Vector]:
        diff.scale_to_length(self.size)
        left_front = diff.rotate(135) + self.pos
        right_front = diff.rotate(-135) + self.pos
        return left_front, right_front

    def update(self, target: Vector, diff: Vector) -> tuple[Vector, Vector]:
        self.pos = target
        diff.scale_to_length(self.size)
        left = Vector(self.pos.x - diff.y, self.pos.y + diff.x)
        right = Vector(self.pos.x + diff.y, self.pos.y - diff.x)
        return left, right

    def draw(self, surf: pygame.surface.Surface):
        pygame.draw.circle(surf, "white", self.pos, self.size, 5)


class Snake:
    def __init__(self):
        self.speed = 1
        self.distance_constraint = 20
        pos = (400, 400)
        sizes = [25 * 0.98**i for i in range(50)]
        sizes[1] = 30
        self.nodes = [
            Node((pos[0] + i * self.distance_constraint, pos[1]), sizes[i])
            for i in range(len(sizes))
        ]

    def update(self, dt: float, target: tuple[int, int]):
        # Head Node
        diff = Vector(target) - self.nodes[0].pos
        target = self.nodes[0].pos + diff * dt * self.speed
        self.left_points = []
        self.right_points = []
        points = self.nodes[0].update(target, -diff)
        front_points = self.nodes[0].calc_front(-diff)
        self.left_points.append(front_points[0])
        self.right_points.append(front_points[1])
        self.left_points.append(points[0])
        self.right_points.append(points[1])

        # Follow Nodes
        for i, node in enumerate(self.nodes[1:]):
            diff = node.pos - self.nodes[i].pos
            diff.scale_to_length(self.distance_constraint)
            target = self.nodes[i].pos + diff
            points = node.update(target, diff)
            self.left_points.append(points[0])
            self.right_points.append(points[1])

    def draw(self, surf: pygame.surface.Surface):
        for node in self.nodes:
            node.draw(surf)
        # bezier(surf, self.left_points, 5, (255, 0, 0))
        # bezier(surf, self.right_points, 5, (0, 255, 0))
        pygame.draw.line(surf, "red", self.left_points[0], self.right_points[0], 3)
        for i in range(len(self.left_points[:-1])):
            pygame.draw.line(
                surf, "red", self.left_points[i], self.left_points[i + 1], 3
            )
            pygame.draw.line(
                surf, "red", self.right_points[i], self.right_points[i + 1], 3
            )
