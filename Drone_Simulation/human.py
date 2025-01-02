import pygame
import constants
import random

class Human:
    def __init__(self, x, y, size=4, color=constants.RED):
        self.x = x
        self.y = y
        self.size = size
        self.color = color

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (self.x, self.y), self.size)
        