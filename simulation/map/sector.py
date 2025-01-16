import pygame
import numpy as np
import pygame
import numpy as np
from simulation.consts.constants import GREEN, WHITE
from simulation.consts.equation_const import ALPHA, BETA




class Sector:
    def __init__(self, row, col, size, searched=False):
        self.row = row
        self.col = col
        self.size = size
        self.searched = searched
        self.attractiveness = 1.0  # Domyślna atrakcyjność
        self.visits = 0  # Licznik odwiedzin
        self.neighbors = []

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    def mark_visited(self):
        self.searched = True
        self.visits += 1
        self.calculate_attractiveness()

    def calculate_attractiveness(self):
        decay_factor = 0.95  # Możesz dostosować współczynnik zaniku
        self.attractiveness = max(0, self.attractiveness * (
                    decay_factor ** self.visits))  # Zanik atrakcyjności z każdą wizytą

        # Obliczenie dodatkowego wpływu sąsiednich nieprzeszukanych sektorów
        neighbor_contribution = ALPHA * sum(
            (1 / (1 + np.linalg.norm(np.array([self.row, self.col]) - np.array([neighbor.row, neighbor.col])))) * (
                1 if not neighbor.searched else 0)
            for neighbor in self.neighbors
        )

        # Dodajemy wpływ sąsiednich sektorów do bieżącej atrakcyjności
        self.attractiveness += neighbor_contribution

        # Karanie za to, że sektor został przeszukany
        penalty = BETA * (1 if self.searched else 0)
        self.attractiveness -= penalty

    def draw(self, screen):
        color = GREEN if self.searched else WHITE
        pygame.draw.rect(
            screen,
            color,
            pygame.Rect(
                self.col * self.size, self.row * self.size, self.size, self.size
            ),
            1  # Grubość linii siatki
        )

