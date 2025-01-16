import unittest
import numpy as np
from simulation.entities.swarm.control_rules import (
    SimpleSeparationRule,
    AlignmentRule,
    CohesionRule,
    AvoidWallsRule,
    AvoidObstaclesRule,
    MoveRightRule,
    SideBySideFormationRule,
)
from simulation.entities.swarm.drones import Drone
from simulation.game_settings import GameSettings
from simulation.utils.vector2D import Vector2D


class TestBoidRules(unittest.TestCase):
    def setUp(self):
        self.game_settings = GameSettings()
        self.drone = Drone(
            pos=np.array([50, 50]),
            game_settings=self.game_settings,
            flock=None,
            rules=[],
            size=10,
        )
        self.local_drones = [
            Drone(pos=np.array([55, 55]), game_settings=self.game_settings, flock=None, rules=[]),
            Drone(pos=np.array([60, 50]), game_settings=self.game_settings, flock=None, rules=[]),
            Drone(pos=np.array([45, 60]), game_settings=self.game_settings, flock=None, rules=[]),
        ]

    def test_simple_separation_rule(self):
        rule = SimpleSeparationRule(weighting=1.0, game_settings=self.game_settings, push_force=10)
        output = rule.evaluate(self.drone, self.local_drones)
        self.assertTrue(output.magnitude() > 0, "Separation rule should produce a non-zero vector.")

    def test_alignment_rule(self):
        for drone in self.local_drones:
            drone.v = Vector2D(1, 0)
        rule = AlignmentRule(weighting=1.0, game_settings=self.game_settings)
        output = rule.evaluate(self.drone, self.local_drones)
        expected_output = Vector2D(1, 0).normalized()
        self.assertAlmostEqual(output.x, expected_output.x, delta=1e-5)
        self.assertAlmostEqual(output.y, expected_output.y, delta=1e-5)

    def test_cohesion_rule(self):
        rule = CohesionRule(weighting=1.0, game_settings=self.game_settings)
        output = rule.evaluate(self.drone, self.local_drones)
        expected_center = np.array([53.3333333, 55.0])
        expected_output = Vector2D(*(expected_center - self.drone.pos)).normalized()
        self.assertAlmostEqual(output.x, expected_output.x, delta=1e-5)
        self.assertAlmostEqual(output.y, expected_output.y, delta=1e-5)

    def test_avoid_walls_rule(self):
        rule = AvoidWallsRule(weighting=1.0, game_settings=self.game_settings, push_force=10)
        self.drone.pos = np.array([5, 5])  # Near the top-left corner
        output = rule.evaluate(self.drone, self.local_drones)
        self.assertTrue(output.magnitude() > 0, "Avoid walls rule should produce a non-zero vector.")

    def test_avoid_obstacles_rule(self):
        class MockObstacle:
            def __init__(self, center):
                self.center = np.array(center)

            def contains(self, pos):
                return np.linalg.norm(pos - self.center) < 10

            def closest_point(self, pos):
                direction = pos - self.center
                return self.center + direction / np.linalg.norm(direction) * 10

        obstacles = [MockObstacle(center=[60, 60]), MockObstacle(center=[20, 20])]
        rule = AvoidObstaclesRule(weighting=1.0, game_settings=self.game_settings, obstacles=obstacles, push_force=10)
        output = rule.evaluate(self.drone, self.local_drones)
        self.assertTrue(output.magnitude() > 0, "Avoid obstacles rule should produce a non-zero vector.")

    def test_move_right_rule(self):
        rule = MoveRightRule(weighting=1.0, game_settings=self.game_settings)
        output = rule.evaluate(self.drone, self.local_drones)
        expected_output = Vector2D(10, 0)
        self.assertAlmostEqual(output.x, expected_output.x)
        self.assertAlmostEqual(output.y, expected_output.y)

    def test_side_by_side_formation_rule(self):
        rule = SideBySideFormationRule(
            weighting=1.0, game_settings=self.game_settings, spacing=10, noise_factor=0.0
        )
        local_drones = [
            Drone(pos=np.array([40, 50]), game_settings=self.game_settings, flock=None, rules=[]),
            Drone(pos=np.array([60, 50]), game_settings=self.game_settings, flock=None, rules=[]),
        ]
        output = rule.evaluate(self.drone, local_drones)
        self.assertTrue(output.magnitude() > 0, "Side by side formation rule should produce a non-zero vector.")

if __name__ == "__main__":
    unittest.main()
