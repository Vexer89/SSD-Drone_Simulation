from typing import List
import logging
import random

import pygame
import numpy as np

#from controls import default_controls
from engine import CharacterEntity
from boids import BoidFlock, BoidRule, SimpleSeparationRule, AvoidWallsRule, AlignmentRule,CohesionRule, SideBySideFormationRule, AvoidObstaclesRule
from game_settings import GameSettings
import sys
from obstacle import *
from human import Human


logging.basicConfig()
logger = logging.getLogger(__name__)


pygame.init()

def parameter_selection_screen():
    pygame.display.set_caption("Parameter Selection - Boid Simulation")
    screen = pygame.display.set_mode((800, 600))
    font = pygame.font.Font(None, 36)
    clock = pygame.time.Clock()

    # Default parameter values
    parameters = {
        "n_boids": 50,
        "n_humans": 15,
        "boid_fear": 20,
        "boid_radius": 100,
        "boid_max_speed": 100
    }

    input_boxes = {
        key: pygame.Rect(400, 100 + i * 50, 140, 40)
        for i, key in enumerate(parameters.keys())
    }
    active_box = None
    user_inputs = {key: str(value) for key, value in parameters.items()}

    def draw_screen():
        screen.fill((50, 50, 50))
        title = font.render("Set Parameters for Boid Simulation", True, (255, 255, 255))
        screen.blit(title, (200, 20))

        for i, (key, rect) in enumerate(input_boxes.items()):
            label = font.render(f"{key}: ", True, (255, 255, 255))
            screen.blit(label, (150, rect.y + 5))
            pygame.draw.rect(screen, (200, 200, 200), rect, 2)
            text_surface = font.render(user_inputs[key], True, (255, 255, 255))
            screen.blit(text_surface, (rect.x + 5, rect.y + 5))

        # Start and Reset buttons
        start_button = pygame.Rect(200, 400, 150, 50)
        pygame.draw.rect(screen, (0, 255, 0), start_button)
        start_text = font.render("Start", True, (0, 0, 0))
        screen.blit(start_text, (start_button.x + 40, start_button.y + 10))

        reset_button = pygame.Rect(400, 400, 150, 50)
        pygame.draw.rect(screen, (255, 0, 0), reset_button)
        reset_text = font.render("Reset", True, (0, 0, 0))
        screen.blit(reset_text, (reset_button.x + 40, reset_button.y + 10))

        return start_button, reset_button

    while True:
        start_button, reset_button = draw_screen()
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                # Activate the clicked input box
                for key, rect in input_boxes.items():
                    if rect.collidepoint(event.pos):
                        active_box = key
                        break
                else:
                    active_box = None

                # Check if buttons were clicked
                if start_button.collidepoint(event.pos):
                    # Return parsed parameters
                    try:
                        return {key: int(value) for key, value in user_inputs.items()}
                    except ValueError:
                        print("Invalid input: Ensure all values are integers.")
                        continue
                if reset_button.collidepoint(event.pos):
                    # Reset to default values
                    user_inputs = {key: str(value) for key, value in parameters.items()}

            if event.type == pygame.KEYDOWN and active_box:
                # Edit the active input box
                if event.key == pygame.K_BACKSPACE:
                    user_inputs[active_box] = user_inputs[active_box][:-1]
                elif event.unicode.isdigit():
                    user_inputs[active_box] += event.unicode

        clock.tick(30)


def check_collision(entity1, entity2, radius):
    entity2_x, entity2_y = entity2.pos
    distance = ((entity1.x - entity2_x) ** 2 + (entity1.y - entity2_y) ** 2) ** 0.5
    return distance < radius

def is_valid_position(x, y, obstacles):
    for obstacle in obstacles:
        if obstacle.contains((x, y)):
            return False
    return True

def main():
    while True:
        parameters = parameter_selection_screen()

        # Extract parameters
        n_boids = parameters["n_boids"]
        n_humans = parameters["n_humans"]
        boid_fear = parameters["boid_fear"]
        boid_radius = parameters["boid_radius"]
        boid_max_speed = parameters["boid_max_speed"]

        game_settings = GameSettings()
        # game_settings.debug = True

        pygame.display.set_caption("Boid Simulation")
        win = pygame.display.set_mode((game_settings.window_width, game_settings.window_height))
        fill_colour = (0, 0, 0)
        light_gray = (200, 200, 200)

        human_positions = [(random.randint(0, win.get_width()), random.randint(0, win.get_height())) for _ in range(n_humans)]
        humans = [Human(x, y) for x, y in human_positions]

        rect1 = RectangleObstacle(100, 100, 300, 100, light_gray)
        rect2 = RectangleObstacle(550, 100, 100, 300, light_gray)
        circ1 = CircleObstacle(700, 200, 100, light_gray)
        circ2 = CircleObstacle(950, 600, 120, light_gray)
        polyg1 = PolygonObstacle([(200, 400), (350, 450), (300, 500), (150, 450)], light_gray)
        polyg2 = PolygonObstacle([(600, 400), (750, 450), (700, 500), (550, 450)], light_gray)
        polyg3 = PolygonObstacle([(400, 600), (550, 650), (500, 700), (350, 650)], light_gray)
        polyg4 = PolygonObstacle([(800, 600), (950, 650), (900, 700), (750, 650)], light_gray)

        obstacles = [rect1, rect2, circ1, circ2, polyg1, polyg3, polyg4]

        human_positions = []
        for _ in range(n_humans):
            while True:
                x, y = random.randint(0, win.get_width()), random.randint(0, win.get_height())
                if is_valid_position(x, y, obstacles):
                    human_positions.append((x, y))
                    break

        humans = [Human(x, y) for x, y in human_positions]


        flock = BoidFlock(game_settings)
        flock_rules: List[BoidRule] = [
            CohesionRule(weighting=0.7, game_settings=game_settings),
            AlignmentRule(weighting=0.7, game_settings=game_settings),
            AvoidWallsRule(weighting=1, game_settings=game_settings, push_force=100),
            SimpleSeparationRule(weighting=0.9, game_settings=game_settings, push_force=boid_fear),
            SideBySideFormationRule(weighting=0.15, game_settings=game_settings, spacing=90, noise_factor=0.1),
            AvoidObstaclesRule(weighting=1, game_settings=game_settings, obstacles=obstacles, push_force=100),
        ]

        flock.generate_boids(n_boids, rules=flock_rules, local_radius=boid_radius, max_velocity=boid_max_speed)

        entities = flock.boids
        tick_length = int(1000/game_settings.ticks_per_second)

        last_tick = pygame.time.get_ticks()
        while game_settings.is_running:
            win.fill(fill_colour)

            for obstacle in obstacles:
                obstacle.draw(win)

            humans = [human for human in humans if not any(check_collision(human, boid, 15) for boid in entities)]

            for i in range(len(humans)):
                Human.draw(humans[i], win)


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
