import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import pyfid.simulation


class Test(unittest.TestCase):
    def test_dummy(self):
        self.assertEqual(1, 1)

class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.s = pyfid.simulation.const_frequency_const_amplitude(
            f0=10,
            sampling_rate=100,
            duration=1,
            snr=10)

    def test_simulation_creation(self):
        self.assertIsInstance(self.s, pyfid.simulation.FIDsim)

    def test_simulation_simulation(self):
        d = self.s.simulate()
        self.assertIsInstance(d, np.ndarray)

    def test_simulation_simulation_length(self):
        d = self.s.simulate()
        self.assertEqual(d.shape[-1], int(self.s.sampling_rate * self.s.duration))

    def test_simulation_multiple(self):
        d = self.s.simulate(n=3)
        self.assertEqual(len(d.shape), 2)
        self.assertEqual(d.shape[0], 3)


if __name__ == '__main__':
    unittest.main()
