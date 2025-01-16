import unittest
import numpy as np
from simulation.utils.vector2D import Vector2D

class TestVector2D(unittest.TestCase):
    def test_initialization(self):
        v = Vector2D(3, 4)
        self.assertEqual(v.x, 3)
        self.assertEqual(v.y, 4)
        self.assertTrue(np.array_equal(v, [3, 4]))

    def test_setters(self):
        v = Vector2D(1, 2)
        v.x = 5
        v.y = 6
        self.assertEqual(v.x, 5)
        self.assertEqual(v.y, 6)

    def test_magnitude(self):
        v = Vector2D(3, 4)
        self.assertEqual(v.magnitude(), 5)

    def test_normalized(self):
        v = Vector2D(3, 4)
        normalized = v.normalized()
        self.assertAlmostEqual(normalized.x, 0.6)
        self.assertAlmostEqual(normalized.y, 0.8)
        self.assertAlmostEqual(normalized.magnitude(), 1)

        # Test dla wektora zerowego
        zero_vector = Vector2D(0, 0)
        self.assertEqual(zero_vector.normalized(), Vector2D(0, 0))

    def test_addition(self):
        v1 = Vector2D(1, 2)
        v2 = Vector2D(3, 4)
        result = v1 + v2
        self.assertEqual(result, Vector2D(4, 6))

    def test_subtraction(self):
        v1 = Vector2D(5, 7)
        v2 = Vector2D(2, 3)
        result = v1 - v2
        self.assertEqual(result, Vector2D(3, 4))

    def test_scalar_multiplication(self):
        v = Vector2D(2, 3)
        result = v * 3
        self.assertEqual(result, Vector2D(6, 9))

    def test_inplace_scalar_multiplication(self):
        v = Vector2D(2, 3)
        v *= 2
        self.assertEqual(v, Vector2D(4, 6))

    def test_inplace_addition(self):
        v1 = Vector2D(1, 1)
        v2 = Vector2D(2, 3)
        v1 += v2
        self.assertEqual(v1, Vector2D(3, 4))

    def test_repr_and_str(self):
        v = Vector2D(1.23456, 7.89012)
        self.assertEqual(repr(v), "Vector2D(1.23, 7.89)")
        self.assertEqual(str(v), "(1.23, 7.89)")

if __name__ == "__main__":
    unittest.main()
