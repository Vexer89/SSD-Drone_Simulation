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
    title_font = pygame.font.Font(None, 48)
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
        key: pygame.Rect(400, 120 + i * 60, 200, 40)
        for i, key in enumerate(parameters.keys())
    }
    active_box = None
    user_inputs = {key: str(value) for key, value in parameters.items()}

    def draw_button(rect, text, color, text_color):
        pygame.draw.rect(screen, color, rect, border_radius=10)
        text_surface = font.render(text, True, text_color)
        text_rect = text_surface.get_rect(center=rect.center)
        screen.blit(text_surface, text_rect)

    def draw_screen():

        screen.fill((30, 30, 30))  # Dark background for a professional look
        title = title_font.render("Parameter Selection", True, (255, 255, 255))
        screen.blit(title, (screen.get_width() // 2 - title.get_width() // 2, 30))

        for i, (key, rect) in enumerate(input_boxes.items()):
            label = font.render(f"{key.replace('_', ' ').capitalize()}: ", True, (200, 200, 200))
            screen.blit(label, (100, rect.y + 5))

            # Highlight active box in dark green
            if key == active_box:
                pygame.draw.rect(screen, (0, 100, 0), rect, border_radius=5)  # Green background for active box
            else:
                pygame.draw.rect(screen, (100, 100, 100), rect, border_radius=5)  # Default background color

            pygame.draw.rect(screen, (200, 200, 200), rect, 2, border_radius=5)  # Border for all boxes
            text_surface = font.render(user_inputs[key], True, (255, 255, 255))
            screen.blit(text_surface, (rect.x + 10, rect.y + 5))

        # Buttons
        start_button = pygame.Rect(200, 500, 150, 50)
        reset_button = pygame.Rect(450, 500, 150, 50)
        draw_button(start_button, "Start", (0, 200, 0), (0, 0, 0))
        draw_button(reset_button, "Reset", (200, 0, 0), (255, 255, 255))

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

        max_width = win.get_width()
        max_height = win.get_height()

        rect1 = RectangleObstacle(random.randint(0, max_width), random.randint(0, max_height), 300, 100, light_gray)
        rect2 = RectangleObstacle(random.randint(0, max_width), random.randint(0, max_height), 100, 300, light_gray)
        circ1 = CircleObstacle(random.randint(0, max_width), random.randint(0, max_height), 100, light_gray)
        circ2 = CircleObstacle(random.randint(0, max_width), random.randint(0, max_height), 120, light_gray)
        # Define the dimensions of the polygons
        polygon1_vertices = [(0, 0), (150, 50), (100, 100), (-50, 50)]
        polygon2_vertices = [(0, 0), (100, 50), (50, 100), (-50, 50)]
        polygon3_vertices = [(0, 0), (200, 100), (150, 200), (-100, 100)]
        polygon4_vertices = [(0, 0), (250, 150), (200, 300), (-150, 150)]

        # Generate random positions for the polygons
        polyg1_origin = (random.randint(0, max_width), random.randint(0, max_height))
        polyg2_origin = (random.randint(0, max_width), random.randint(0, max_height))
        polyg3_origin = (random.randint(0, max_width), random.randint(0, max_height))
        polyg4_origin = (random.randint(0, max_width), random.randint(0, max_height))

        # Create the polygons with random positions
        polyg1 = PolygonObstacle([(x + polyg1_origin[0], y + polyg1_origin[1]) for x, y in polygon1_vertices], light_gray)
        polyg2 = PolygonObstacle([(x + polyg2_origin[0], y + polyg2_origin[1]) for x, y in polygon2_vertices], light_gray)
        polyg3 = PolygonObstacle([(x + polyg3_origin[0], y + polyg3_origin[1]) for x, y in polygon3_vertices], light_gray)
        polyg4 = PolygonObstacle([(x + polyg4_origin[0], y + polyg4_origin[1]) for x, y in polygon4_vertices], light_gray)

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

        flock.generate_boids(n_boids, rules=flock_rules, local_radius=boid_radius, max_velocity=boid_max_speed, obstacles=obstacles)
        
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
