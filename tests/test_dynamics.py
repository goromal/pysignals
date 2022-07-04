import pytest
import numpy as np
from pysignals import *
from geometry import SO3, SE3

class Helpers:
    @staticmethod
    def test_dynamics(uType, sysType, stateType, xd, prepFunc, checkFunc):
        sys = sysType()
        sys2 = sysType()
        
        prepFunc(sys)
        prepFunc(sys2)
        
        u = uType()

        t = 0
        kp = 1
        kd = 0.5

        assert sys.x.update(t, stateType.identity())
        assert sys2.x.update(t, stateType.identity())

        dt = 0.01
        num_iters = 10000

        for _ in range(num_iters):
            assert u.update(t, kp * (xd - sys.x().pose) - kd * sys.x().twist, True)
            t += dt
            assert sys.simulateEuler(u, t)

        checkFunc(sys, xd)
        assert sys2.simulateEuler(u, t, dt)
        checkFunc(sys2, xd)

@pytest.fixture
def helpers():
    return Helpers

class TestDynamics:
    def test_trans_dynamics(self, helpers):
        xd = 1
        def prepFunc(sys):
            sys.dynamics.g = 0
        def checkFunc(sys, xd):
            assert np.allclose(sys.x().pose, xd)
            assert np.allclose(sys.x().twist, 0)
        helpers.test_dynamics(ScalarSignal, Translational1DOFSystem, ScalarState, xd, prepFunc, checkFunc)

    def test_so2_dynamics(self, helpers):
        xd = SO2.fromAngle(1.0)
        def prepFunc(sys):
            pass
        def checkFunc(sys, xd):
            assert np.allclose(sys.x().pose.array(), xd.array())
            assert np.allclose(np.linalg.norm(sys.x().twist), 0)
        helpers.test_dynamics(Vector1Signal, Rotational1DOFSystem, SO2State, xd, prepFunc, checkFunc) 

    def test_so3_dynamics(self, helpers):
        xd = SO3.fromEuler(1.0, -2.0, 0.5)
        def prepFunc(sys):
            pass
        def checkFunc(sys, xd):
            assert np.allclose(sys.x().pose.array(), xd.array())
            assert np.allclose(np.linalg.norm(sys.x().twist), 0)
        helpers.test_dynamics(Vector3Signal, Rotational3DOFSystem, SO3State, xd, prepFunc, checkFunc)

    def test_se3_dynamics(self, helpers):
        xd = SE3.fromVecAndQuat(np.array([0.5, -3.0, 2.0]), SO3.fromEuler(1.0, -2.0, 0.5))
        def prepFunc(sys):
            pass
        def checkFunc(sys, xd):
            assert np.allclose(sys.x().pose.array(), xd.array())
            assert np.allclose(np.linalg.norm(sys.x().twist), 0)
        helpers.test_dynamics(Vector6Signal, RigidBody6DOFSystem, SE3State, xd, prepFunc, checkFunc)
