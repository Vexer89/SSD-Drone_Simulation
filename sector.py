import pygame
import numpy as np
from constants import *
from equation_const import ALPHA, BETA

class Sector:
    def __init__(self, row, col, size, searched=False):
        self.row = row
        self.col = col
        self.size = size
        self.searched = searched
        self.attractiveness = 1.0  # Default attractiveness
        self.neighbors = []

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)
    
    def calculate_attractiveness(self):
        neighbor_contribution = ALPHA * sum(
            (1 / (1 + np.linalg.norm(np.array([self.row, self.col]) - np.array([neighbor.row, neighbor.col])))) * (1 if not neighbor.searched else 0)
            for neighbor in self.neighbors
        )
        penalty = BETA * (1 if self.searched else 0)
        self.attractiveness = neighbor_contribution - penalty

    def draw(self, screen):
        color = GREEN if self.searched else GRAY
        pygame.draw.rect(
            screen,
            color,
            pygame.Rect(
                self.col * self.size, self.row * self.size, self.size, self.size
            ),
            1  # Line thickness for grid
        )

    def mark_searched(self):
        self.searched = True

    def contains_point(self, point):
        """
        Sprawdza, czy dany punkt (x, y) znajduje się w sektorze.
        """
        x, y = point
        x_min = self.col * self.size
        x_max = x_min + self.size
        y_min = self.row * self.size
        y_max = y_min + self.size

        return x_min <= x < x_max and y_min <= y < y_max
