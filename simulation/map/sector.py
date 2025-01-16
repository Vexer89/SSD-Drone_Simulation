import pygame
import numpy as np
from consts.constants import GREEN, WHITE
from consts.equation_const import ALPHA, BETA

#TODO:
# maybe sector should inherit entity?
# something to check which drones are inside

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
        color = GREEN if self.searched else WHITE
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
