import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import pyfid.simulation
import pyfid.estimation
import pyfid.nEDMatPSI


# traceback for warnings
# def setUpModule():
#     import warnings
#     warnings.simplefilter("error")


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


class TestEstimationMisc(unittest.TestCase):
    def test_divide_for_periods(self):
        d = np.sin(np.linspace(0, 10, 100))
        iCrossings = pyfid.estimation.divide_for_periods(d)
        self.assertIsInstance(iCrossings, np.ndarray)


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
        f, sf, _details = pyfid.estimation.direct_fit(
            T=self.sim.T,
            D=self.sim.simulate(),
            sD=self.sim.sigma(),
            model_key="double_damped_sine_DC")
        self.assertGreater(5 * sf, np.abs(f - self.sim.real_favg()))

    def test_two_windows(self):
        f, sf, _details = pyfid.estimation.two_windows(
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
        f, sf, _details = pyfid.estimation.direct_fit(
            T=self.sim.T,
            D=self.sim.simulate(),
            sD=self.sim.sigma(),
            model_key="double_damped_sine_DC")
        self.assertGreater(5 * sf, np.abs(f - self.sim.real_favg()))

    def test_two_windows(self):
        f, sf, _details = pyfid.estimation.two_windows(
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
        f, sf, _details = pyfid.estimation.direct_fit(
            T=self.sim.T,
            D=self.sim.simulate(),
            sD=self.sim.sigma(),
            model_key="double_damped_sine_DC")
        self.assertGreater(5 * sf, np.abs(f - self.sim.real_favg()))

    def test_two_windows(self):
        f, sf, _details = pyfid.estimation.two_windows(
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


class TestOptimization(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.sim_gen = lambda: pyfid.simulation.rand_poly_frequency_two_exp_amplitude(
            f0=7.8,
            t1=14.3,
            t2=1.2,
            t1_to_t2_amplitudes_ratio=0.1,
            deg=1,
            drift=0.1,
            duration=18.0,
            fs=100,
            snr=200)

    def test_accuracy_and_precision(self):
        direct_fit_estimator = lambda T, D, sD: pyfid.estimation.direct_fit(
            T, D, sD, model_key="double_damped_sine_DC")

        pyfid.optimization.accuracy_and_precision_different_sims(
                sim_gen=self.sim_gen,
                estimator=direct_fit_estimator,
                nsimulations=5,
                nsignals=5)

    def test_scan(self):
        win_lengths = np.linspace(2, 12, num=10)

        for win_length in win_lengths:
            estimator = lambda T, D, sD: pyfid.estimation.two_windows(
                T=T, D=D, sD=sD,
                submethod='phase',
                prenormalize=False,
                double_exp=(True, False),
                phase_at_end=True,
                win_len=(win_length / 20, win_length),
                verbose=False)

            pyfid.optimization.accuracy_and_precision_different_sims(
                sim_gen=self.sim_gen,
                estimator=estimator,
                nsimulations=5,
                nsignals=5,
                full_output=True)

    def test_bisection(self):
        estimator = lambda p, T, D, sD: pyfid.estimation.two_windows(
            T=T, D=D, sD=sD,
            submethod='phase',
            prenormalize=False,
            double_exp=(True, False),
            phase_at_end=True,
            win_len=(p / 20, p),
            verbose=False)

        _optimum = pyfid.optimization.bisect_parameter(
            sim_gen=self.sim_gen,
            estimator=estimator,
            p_min=2,
            p_max=12,
            p_tol=1,
            nsimulations=10,
            nsignals=10)

    def test_bisection_conservative(self):
        estimator = lambda p, T, D, sD: pyfid.estimation.two_windows(
            T=T, D=D, sD=sD,
            submethod='phase',
            prenormalize=False,
            double_exp=(True, False),
            phase_at_end=True,
            win_len=(p / 20, p),
            verbose=False)

        _optimum = pyfid.optimization.bisect_parameter(
            sim_gen=self.sim_gen,
            estimator=estimator,
            p_min=2,
            p_max=12,
            p_tol=1,
            nsimulations=10,
            nsignals=10,
            sigmas=1)



if __name__ == '__main__':
    unittest.main()
