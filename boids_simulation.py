from typing import List
import logging

import pygame
import numpy as np

#from controls import default_controls
from drones import DroneFlock
from control_rules import BoidRule, SimpleSeparationRule, AvoidWallsRule, AlignmentRule,CohesionRule, SideBySideFormationRule, AvoidObstaclesRule
from game_settings import GameSettings

from obstacle import *


logging.basicConfig()
logger = logging.getLogger(__name__)


pygame.init()

#TODO:
# start button
# sector and human generation
# human mark as found
# simulation  time limit

def main():
    game_settings = GameSettings()
    # game_settings.debug = True

    pygame.display.set_caption("Drone Simulation")
    win = pygame.display.set_mode((game_settings.window_width, game_settings.window_height))
    fill_colour = (0, 0, 0)
    light_gray = (200, 200, 200)

    n_boids = 50
    boid_fear = 20
    boid_radius = 100
    boid_max_speed = 100

    rect1 = RectangleObstacle(100, 100, 300, 100, light_gray)
    rect2 = RectangleObstacle(550, 100, 100, 300, light_gray)
    circ1 = CircleObstacle(700, 200, 100, light_gray)
    circ2 = CircleObstacle(950, 600, 120, light_gray)
    polyg1 = PolygonObstacle([(200, 400), (350, 450), (300, 500), (150, 450)], light_gray)
    polyg2 = PolygonObstacle([(600, 400), (750, 450), (700, 500), (550, 450)], light_gray)
    polyg3 = PolygonObstacle([(400, 600), (550, 650), (500, 700), (350, 650)], light_gray)
    polyg4 = PolygonObstacle([(800, 600), (950, 650), (900, 700), (750, 650)], light_gray)

    obstacles = [rect1, rect2, circ1, circ2, polyg1, polyg3, polyg4]

    flock = DroneFlock(game_settings)
    flock_rules: List[BoidRule] = [
        CohesionRule(weighting=0.7, game_settings=game_settings),
        AlignmentRule(weighting=1, game_settings=game_settings),
        AvoidWallsRule(weighting=1, game_settings=game_settings, push_force=100),
        SimpleSeparationRule(weighting=1, game_settings=game_settings, push_force=boid_fear),
        SideBySideFormationRule(weighting=0.25, game_settings=game_settings, spacing=90, noise_factor=0.1),
        AvoidObstaclesRule(weighting=1, game_settings=game_settings, obstacles=obstacles, push_force=100),
    ]

    flock.generate_drones(n_boids, rules=flock_rules, local_radius=boid_radius, max_velocity=boid_max_speed)

    entities = flock.drones
    tick_length = int(1000/game_settings.ticks_per_second)

    last_tick = pygame.time.get_ticks()
    while game_settings.is_running:
        win.fill(fill_colour)

        for obstacle in obstacles:
            obstacle.draw(win)

        time_since_last_tick = pygame.time.get_ticks() - last_tick
        if time_since_last_tick < tick_length:
            pygame.time.delay(tick_length - time_since_last_tick)

        time_since_last_tick = pygame.time.get_ticks() - last_tick

        last_tick = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_settings.is_running = False

        keys = pygame.key.get_pressed()

        for entity in entities:
            entity.update(keys, win, time_since_last_tick/1000)

        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
