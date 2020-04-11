import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import pyfid.simulation
import pyfid.estimation
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


class TestMethodsNoDrift(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

        self.sim = pyfid.simulation.const_frequency_two_exp_amplitude(
            fs=pyfid.nEDMatPSI.fs,
            f0=pyfid.nEDMatPSI.filter_f0,
            duration=pyfid.nEDMatPSI.duration,
            snr=144,
            t1=pyfid.nEDMatPSI.t1,
            t2=pyfid.nEDMatPSI.t2,
            t1_to_t2_amplitudes_ratio=pyfid.nEDMatPSI.t1_to_t2_amplitudes_ratio)

    def test_direct_fit(self):
        f, sf, details = pyfid.estimation.direct_fit(
            T=self.sim.T,
            D=self.sim.simulate(),
            sD=self.sim.sigma(),
            double_exp=True)
        self.assertGreater(5 * sf, np.abs(f - self.sim.real_favg()))

    def test_two_windows(self):
        f, sf, details = pyfid.estimation.two_windows(
            T=self.sim.T,
            D=self.sim.simulate(),
            sD=self.sim.sigma(),
            submethod='phase',
            prenormalize=False,
            double_exp=(True, False),
            phase_at_end=True,
            win_len=(1, 3),
            verbose=False)
        self.assertGreater(5 * sf, np.abs(f - self.sim.real_favg()))


class TestMethodsFilter(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

        self.sim = pyfid.simulation.const_frequency_two_exp_amplitude(
            fs=pyfid.nEDMatPSI.fs,
            f0=pyfid.nEDMatPSI.filter_f0,
            duration=pyfid.nEDMatPSI.duration,
            snr=144,
            t1=pyfid.nEDMatPSI.t1,
            t2=pyfid.nEDMatPSI.t2,
            t1_to_t2_amplitudes_ratio=pyfid.nEDMatPSI.t1_to_t2_amplitudes_ratio,
            filter_func=pyfid.nEDMatPSI.nEDMfilter)

    def test_direct_fit(self):
        f, sf, details = pyfid.estimation.direct_fit(
            T=self.sim.T,
            D=self.sim.simulate(),
            sD=self.sim.sigma(),
            double_exp=True)
        self.assertGreater(5 * sf, np.abs(f - self.sim.real_favg()))

    def test_two_windows(self):
        f, sf, details = pyfid.estimation.two_windows(
            T=self.sim.T,
            D=self.sim.simulate(),
            sD=self.sim.sigma(),
            submethod='phase',
            prenormalize=False,
            double_exp=(True, False),
            phase_at_end=True,
            win_len=(1, 3),
            verbose=False)
        self.assertGreater(5 * sf, np.abs(f - self.sim.real_favg()))


class TestMethodsFilterAdvanceTime(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

        self.sim = pyfid.simulation.const_frequency_two_exp_amplitude(
            fs=pyfid.nEDMatPSI.fs,
            f0=pyfid.nEDMatPSI.filter_f0,
            duration=pyfid.nEDMatPSI.duration,
            snr=144,
            t1=pyfid.nEDMatPSI.t1,
            t2=pyfid.nEDMatPSI.t2,
            t1_to_t2_amplitudes_ratio=pyfid.nEDMatPSI.t1_to_t2_amplitudes_ratio,
            filter_advance_time=1,
            filter_func=pyfid.nEDMatPSI.nEDMfilter)

    def test_direct_fit(self):
        f, sf, details = pyfid.estimation.direct_fit(
            T=self.sim.T,
            D=self.sim.simulate(),
            sD=self.sim.sigma(),
            double_exp=True)
        self.assertGreater(5 * sf, np.abs(f - self.sim.real_favg()))

    def test_two_windows(self):
        f, sf, details = pyfid.estimation.two_windows(
            T=self.sim.T,
            D=self.sim.simulate(),
            sD=self.sim.sigma(),
            submethod='phase',
            prenormalize=False,
            double_exp=(True, False),
            phase_at_end=True,
            win_len=(1, 3),
            verbose=False)
        self.assertGreater(5 * sf, np.abs(f - self.sim.real_favg()))

if __name__ == '__main__':
    unittest.main()
