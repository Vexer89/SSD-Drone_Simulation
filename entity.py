from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, List

import numpy as np
import pygame

from game_settings import MapEdgeBehaviour, GameSettings
from vector2D import Vector2D


class Entity(ABC):
    def __init__(self, game_settings: GameSettings, pos: Vector2D = Vector2D(0, 0),
                 colour: tuple[int, int, int] = (255, 0, 0), **kwargs):

        self._pos = pos
        self.colour = colour
        self.game_settings: GameSettings = game_settings
        self.kwargs = kwargs

    @abstractmethod
    def draw(self, win):
        pass

    @property
    def pos(self) -> Vector2D:
        return self._pos

    @pos.setter
    def pos(self, pos: Vector2D):
        self._pos = pos

    def distance_to(self, other: 'Entity') -> float:
        return (self.pos - other.pos).magnitude()

    @abstractmethod
    def update_physics(self, time_elapsed):
        pass

    def check_physics(self):
        if self.pos[0] > self.game_settings.map_width:
            if self.game_settings.x_edge_behaviour == MapEdgeBehaviour.WRAP:
                self.pos[0] = self.pos[0] % self.game_settings.map_width
            if self.game_settings.x_edge_behaviour == MapEdgeBehaviour.CLAMP:
                self.pos[0] = self.game_settings.map_width

        if self.pos[0] < 0:
            if self.game_settings.x_edge_behaviour == MapEdgeBehaviour.WRAP:
                self.pos[0] = self.pos[0] % self.game_settings.map_width
            if self.game_settings.x_edge_behaviour == MapEdgeBehaviour.CLAMP:
                self.pos[0] = 0

        if self.pos[1] > self.game_settings.map_width:
            if self.game_settings.y_edge_behaviour == MapEdgeBehaviour.WRAP:
                self.pos[1] = self.pos[1] % self.game_settings.map_height
            if self.game_settings.y_edge_behaviour == MapEdgeBehaviour.CLAMP:
                self.pos[1] = self.game_settings.map_height

        if self.pos[1] < 0:
            if self.game_settings.y_edge_behaviour == MapEdgeBehaviour.WRAP:
                self.pos[1] = self.pos[1] % self.game_settings.map_width
            if self.game_settings.y_edge_behaviour == MapEdgeBehaviour.CLAMP:
                self.pos[1] = 0

    def get_debug_text(self):
        return f"pos:{self.pos[0]:0.1f}, {self.pos[1]:0.1f}"

    def draw_debug_info(self, win):
        font = pygame.font.SysFont("Courier New", 16)
        text_surface = font.render(self.get_debug_text(), True, self.game_settings.debug_text_colour)
        win.blit(text_surface, (self.pos[0], self.pos[1]))

    def update(self, keys, win, time_elapsed):
        self.update_physics(time_elapsed)
        self.check_physics()
        self.draw(win)

        if self.game_settings.debug:
            self.draw_debug_info(win)


