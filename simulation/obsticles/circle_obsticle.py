import random
import pygame

class CircleObstacle:
    def __init__(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)

    def is_point_inside(self, point):
        # Sprawdza, czy punkt jest wewnątrz okręgu
        px, py = point
        return (px - self.x) ** 2 + (py - self.y) ** 2 <= self.radius ** 2

