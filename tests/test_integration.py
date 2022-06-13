import pytest
import numpy as np
from pysignals import *
from geometry import SO3

class Helpers:
    @staticmethod
    def test_integrator(integratorFunc):
        dt = 0.001
        t = 0
        tf = 5. * np.pi / 4.
        v_ref, v_int = ScalarSignal(), ScalarSignal()
        while t <= tf:
            v_ref.update(t, np.sin(t), np.cos(t), True)
            t += dt
        v_ref_dot = v_ref.dotSignal()
        v_int.update(0., 0.)
        integratorFunc(v_int, v_ref_dot, t, dt, True)
        assert abs(v_int(0.) - v_ref(0.)) < 0.01
        assert abs(v_int(np.pi/4.) - v_ref(np.pi/4.)) < 0.01
        assert abs(v_int(np.pi/2.) - v_ref(np.pi/2.)) < 0.01
        assert abs(v_int(3.*np.pi/4.) - v_ref(3.*np.pi/4.)) < 0.01

@pytest.fixture
def helpers():
    return Helpers

class TestIntegration:
    def test_euler_integrator(self, helpers):
        helpers.test_integrator(integrateEuler)

    def test_trapezoidal_integrator(self, helpers):
        helpers.test_integrator(integrateTrapezoidal)
