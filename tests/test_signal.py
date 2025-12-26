import pytest
import numpy as np
from pysignals import *
from geometry import SO3

class TestSignal:
    def test_linear_interpolation(self):
        a = ScalarSignal()
        b = ScalarSignal()
        c = ScalarSignal()

        a.setInterpolationMethod(Interpolation.LINEAR)
        a.setExtrapolationMethod(Extrapolation.ZEROS)
        b.setInterpolationMethod(Interpolation.LINEAR)
        b.setExtrapolationMethod(Extrapolation.ZEROS)
        c.setInterpolationMethod(Interpolation.LINEAR)
        c.setExtrapolationMethod(Extrapolation.ZEROS)

        assert a.update(0., 2., 1., True)
        assert a.update(1., 3., 1., True)
        assert a.update(2., 4., 1., True)

        assert b.update([0., 1., 2.], [2., 3., 4.])

        assert c.update(0., 2., 1., True)
        assert c.update([1., 2.], [3., 4.])

        assert a(1.5) == 3.5
        assert b(1.5) == 3.5
        assert c(1.5) == 3.5

        assert a(0.) == 2.
        assert b(0.) == 2.
        assert c(0.) == 2.

        assert a(2.01) == 0.
        assert b(2.01) == 0.
        assert c(2.01) == 0.

        assert a.dot(1.5) == 1.
        assert c.dot(1.5) == 1.
        assert b.dot(1.5) > 0.

        assert a.dot(2.01) == 0.
        assert c.dot(2.01) == 0.
        assert b.dot(2.01) == 0.

        a.setExtrapolationMethod(Extrapolation.CLOSEST)

        assert a(2.01) == 4.
        assert a.dot(2.01) == 1.

        q = SO3Signal()
        
        q.setInterpolationMethod(Interpolation.LINEAR)
        q.setExtrapolationMethod(Extrapolation.ZEROS)

        assert q.update(0., SO3.fromEuler(0., 0., 0.), True)
        assert q.update(1., SO3.fromEuler(0., 0., np.pi), True)

        q_half = SO3.fromEuler(0., 0., np.pi/2.)
        q_iden = SO3.identity()

        assert np.allclose(q(0.5).array(), q_half.array())
        assert np.allclose(q(-0.5).array(), q_iden.array())
    
    def test_norms(self):
        q = SO3.identity()
        v = np.array([3., 4., 0.])
        
        assert abs(SO3Signal.baseNorm(q)) < 1e-8
        assert abs(SO3Signal.tangentNorm(v) - 5.0) < 1e-8

    def test_set_equality(self):
        v1 = ScalarSignal()
        v1.update(0., 4., True)
        v1.update(10., -10., True)

        v2 = v1

        assert v1(5.) == v2(5.)
        assert v1.dot(5.) == v2.dot(5.)
        assert v1(20.) == v2(20.)
        assert v1.dot(20.) == v2.dot(20.)

    def test_scaled_plus_minus(self):
        v1 = ScalarSignal()
        v2 = ScalarSignal()
        v3 = ScalarSignal()
        dt = 0.01
        for i in range(1000):
            t = i*dt
            v1.update(t, np.sin(t), True)
            v2.update(t, np.cos(t), True)
            v3.update(t, np.sin(2.*t), True)
        u = v1 + 2. * v2 - 3. * v3
        t_test = 5.0004
        assert abs(u(t_test) - (np.sin(t_test) + 2.*np.cos(t_test) - 3.*np.sin(2.*t_test))) < 0.001

    def test_scaled_manifold_plus_minus(self):
        v1 = SO3Signal()
        v2 = SO3Signal()
        v3 = SO3Signal()
        dt = 0.01
        for i in range(1000):
            t = i * dt
            v1.update(t, SO3.fromEuler(0.1, -0.1, 0.), True)
            v2.update(t, SO3.fromEuler(0.2, -0.3, 0.), True)
            v3.update(t, SO3.fromEuler(-0.1, 0.24, 0.1), True)
        u = v1 + (2.*v2 - v3)
        t_test = 5.0004
        q_est = u(t_test)
        q_tru = SO3.fromEuler(0.1, -0.1, 0.) + (2. * SO3.fromEuler(0.2, -0.3, 0.) - SO3.fromEuler(-0.1, 0.24, 0.1))
        assert np.allclose(q_est.array(), q_tru.array())
