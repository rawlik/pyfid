import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import pyfid.simulation
import pyfid.nEDMatPSI


class Test(unittest.TestCase):
    def test_dummy(self):
        self.assertEqual(1, 1)

class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.s = pyfid.simulation.const_frequency_const_amplitude(
            f0=10,
            fs=100,
            duration=1,
            snr=10)

    def test_simulation_creation(self):
        self.assertIsInstance(self.s, pyfid.simulation.FIDsim)

    def test_simulation_simulation(self):
        d = self.s.simulate()
        self.assertIsInstance(d, np.ndarray)

    def test_simulation_simulation_length(self):
        d = self.s.simulate()
        self.assertEqual(d.shape[-1], int(self.s.fs * self.s.duration))

    def test_simulation_multiple(self):
        d = self.s.simulate(n=3)
        self.assertEqual(len(d.shape), 2)
        self.assertEqual(d.shape[0], 3)

    def test_simulation_random_phase(self):
        d = self.s.simulate(n=2, random_phase=True)
        self.assertNotEqual(d[0,0], d[1,0])

class TestFilter(unittest.TestCase):
    def test_nEDMfilter(self):
        n = 100
        f = pyfid.nEDMatPSI.nEDMfilter(np.random.rand(n))
        self.assertIsInstance(f, np.ndarray)
        self.assertEqual(f.shape[0], n)


if __name__ == '__main__':
    unittest.main()
