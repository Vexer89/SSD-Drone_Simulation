import unittest
import numpy as np
from simulation.entities.swarm.drones import Drone, DroneFlock
from simulation.game_settings import GameSettings
from simulation.utils.vector2D import Vector2D


class TestDroneFlock(unittest.TestCase):
    def setUp(self):
        self.game_settings = GameSettings()
        self.game_settings.map_width = 500
        self.game_settings.map_height = 500
        self.flock = DroneFlock(game_settings=self.game_settings)

    def test_generate_drones(self):
        self.flock.generate_drones(10)
        self.assertEqual(len(self.flock.drones), 10, "Flock should generate the correct number of drones.")
        for drone in self.flock.drones:
            self.assertGreaterEqual(drone.pos[0], 0)
            self.assertLessEqual(drone.pos[0], self.game_settings.map_width)
            self.assertGreaterEqual(drone.pos[1], 0)
            self.assertLessEqual(drone.pos[1], self.game_settings.map_height)

    def test_get_local_drones(self):
        self.flock.generate_drones(5)
        drone = self.flock.drones[0]
        drone.local_radius = 200
        local_drones = self.flock.get_local_drones(drone)
        self.assertGreaterEqual(len(local_drones), 0, "Local drones should be non-negative.")
        self.assertLessEqual(len(local_drones), 5, "Local drones should be at most 5.")

    def test_get_local_drones_sorted(self):
        self.flock.generate_drones(5)
        drone = self.flock.drones[0]
        drone.local_radius = 200
        local_drones = self.flock.get_local_drones(drone)
        for i in range(len(local_drones) - 1):
            dist1 = drone.distance_to(local_drones[i])
            dist2 = drone.distance_to(local_drones[i + 1])
            self.assertLessEqual(dist1, dist2, "Local drones should be sorted by distance.")

class TestDrone(unittest.TestCase):
    def setUp(self):
        self.game_settings = GameSettings()
        self.game_settings.map_width = 500
        self.game_settings.map_height = 500
        self.flock = DroneFlock(game_settings=self.game_settings)
        self.drone = Drone(
            pos=np.array([250, 250]),
            game_settings=self.game_settings,
            flock=self.flock,
            size=10,
            local_radius=100,
            max_velocity=30,
            speed=20,
            rules=[],
        )

    def test_velocity_constraint(self):
        self.drone.v = Vector2D(50, 50)
        self.assertLessEqual(
            self.drone.v.magnitude(),
            self.drone.max_velocity,
            "Drone velocity should not exceed the maximum velocity.",
        )

    def test_draw(self):
        """
        Ensure the draw method does not raise errors.
        Since pygame is graphical, we will only validate execution without exceptions.
        """
        import pygame

        pygame.init()
        screen = pygame.display.set_mode((500, 500))
        try:
            self.drone.draw(screen)
        except Exception as e:
            self.fail(f"Draw method raised an exception: {e}")
        finally:
            pygame.quit()

    def test_update_physics(self):
        self.flock.generate_drones(5)
        self.drone.update_physics(1)
        # Check if position is updated correctly
        self.assertNotEqual(
            self.drone.pos.tolist(),
            [250, 250],
            "Drone position should be updated after physics update.",
        )

    def test_calculate_rules_no_rules(self):
        local_drones = []
        acceleration = self.drone.calculate_rules(local_drones)
        self.assertEqual(acceleration.x, 0)
        self.assertEqual(acceleration.y, 0)

    def test_calculate_rules_with_rule(self):
        class MockRule:
            def evaluate(self, drone, local_drones):
                return Vector2D(10, 5)

        self.drone.rules = [MockRule()]
        local_drones = []
        acceleration = self.drone.calculate_rules(local_drones)
        self.assertEqual(acceleration.x, 10)
        self.assertEqual(acceleration.y, 5)

    def test_local_neighborhood_count(self):
        self.flock.generate_drones(5)
        self.drone.update_physics(0.1)
        self.assertGreaterEqual(
            self.drone.n_neighbours, 0, "Neighbor count should be non-negative."
        )


if __name__ == "__main__":
    unittest.main()
