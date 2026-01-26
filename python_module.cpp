#include <signals/Signals.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <sstream>

using namespace Eigen;
namespace py = pybind11;

#define WRAP_SIGNAL_TYPE(SignalName, BST, TST) {\
    py::class_<BST>(m, SignalName)\
    .def_readwrite("interpolationMethod", &BST::interpolationMethod)\
    .def_readwrite("extrapolationMethod", &BST::extrapolationMethod)\
    .def_readwrite("derivativeMethod", &BST::derivativeMethod)\
    .def_static("baseZero", &BST::baseZero)\
    .def_static("tangentZero", &BST::tangentZero)\
    .def_static("baseNorm", &BST::baseNorm)\
    .def_static("tangentNorm", &BST::tangentNorm)\
    .def(py::init())\
    .def(py::init<const BST &>())\
    .def("dotSignal", &BST::dotSignal)\
    .def(py::self + TST())\
    .def(py::self - BST())\
    .def(float() * py::self)\
    .def(py::self * float())\
    .def("t", &BST::t)\
    .def("__call__", static_cast<BST::BaseType (BST::*)(void) const>(&BST::operator()), "Get most recent signal value")\
    .def("__call__", static_cast<BST::BaseType (BST::*)(const double&) const>(&BST::operator()), "Get signal value at time")\
    .def("__call__", static_cast<std::vector<BST::BaseType> (BST::*)(const std::vector<double> &) const>(&BST::operator()), "Get signal values at times")\
    .def("dot", static_cast<BST::TangentType (BST::*)(void) const>(&BST::dot), "Get most recent derivative value")\
    .def("dot", static_cast<BST::TangentType (BST::*)(const double&) const>(&BST::dot), "Get derivative value at time")\
    .def("dot", static_cast<std::vector<BST::TangentType> (BST::*)(const std::vector<double> &) const>(&BST::dot), "Get derivative values at times")\
    .def("setInterpolationMethod", &BST::setInterpolationMethod)\
    .def("setExtrapolationMethod", &BST::setExtrapolationMethod)\
    .def("setDerivativeMethod", &BST::setDerivativeMethod)\
    .def("reset", &BST::reset)\
    .def("update", static_cast<bool (BST::*)(const double&, const BST::BaseType&, bool)>(&BST::update), "Update signal value", py::arg("t"), py::arg("x"), py::arg("insertHistory") = false)\
    .def("update", static_cast<bool (BST::*)(const double&, const BST::BaseType&, const BST::TangentType&, bool)>(&BST::update), "Update signal and derivative value", py::arg("t"), py::arg("x"), py::arg("xdot"), py::arg("insertHistory") = false)\
    .def("update", static_cast<bool (BST::*)(const std::vector<double>&, const std::vector<BST::BaseType>&)>(&BST::update), "Update signal values", py::arg("tHistory"), py::arg("xHistory"))\
    .def("update", static_cast<bool (BST::*)(const std::vector<double>&, const std::vector<BST::BaseType>&, const std::vector<BST::TangentType>&)>(&BST::update), "Update signal and derivative values", py::arg("tHistory"), py::arg("xHistory"), py::arg("xdotHistory"))\
    .def_static("baseZero", &BST::baseZero, "Get zero value for base type")\
    .def_static("tangentZero", &BST::tangentZero, "Get zero value for tangent type")\
    .def_static("baseNorm", &BST::baseNorm, "Compute norm of base type", py::arg("x"))\
    .def_static("tangentNorm", &BST::tangentNorm, "Compute norm of tangent type", py::arg("x"));\
}

#define WRAP_STATE_TYPE(StateName, BST, TST) {\
    py::class_<BST>(m, StateName)\
    .def_readwrite("pose", &BST::pose)\
    .def_readwrite("twist", &BST::twist)\
    .def(py::init())\
    .def(py::init<const BST &>())\
    .def_static("identity", &BST::identity)\
    .def(py::self + TST())\
    .def(py::self - BST())\
    .def(float() * py::self)\
    .def(py::self * float());\
}

#define _WRAP_INTEGRATOR_FOR_SPECS(FuncName, IT, T, BSS, TSS) {\
    m.def(FuncName, static_cast<bool (*)(Signal<T, BSS, TSS>&, const Signal<T, TSS, TSS>&, const double&, const bool&)>(&IT::integrate), "Integrate over the whole interval up to t", py::arg("xInt"), py::arg("x"), py::arg("t"), py::arg("insertIntoHistory") = false);\
    m.def(FuncName, static_cast<bool (*)(Signal<T, BSS, TSS>&, const Signal<T, TSS, TSS>&, const double&, const double&, const bool&)>(&IT::integrate), "Integrate over the whole interval up to t in increments of dt", py::arg("xInt"), py::arg("x"), py::arg("t"), py::arg("dt"), py::arg("insertIntoHistory") = false);\
}

#define SSS ScalarSignalSpec<double>
#define V1S VectorSignalSpec<double, 1>
#define V2S VectorSignalSpec<double, 2>
#define V3S VectorSignalSpec<double, 3>
#define V4S VectorSignalSpec<double, 4>
#define V5S VectorSignalSpec<double, 5>
#define V6S VectorSignalSpec<double, 6>
#define V7S VectorSignalSpec<double, 7>
#define V8S VectorSignalSpec<double, 8>
#define V9S VectorSignalSpec<double, 9>
#define V10S VectorSignalSpec<double, 10>
#define SO3S ManifoldSignalSpec<double, SO3d>
#define SE3S ManifoldSignalSpec<double, SE3d>

#define WRAP_INTEGRATOR_TYPE(FuncName, IT) {\
    _WRAP_INTEGRATOR_FOR_SPECS(FuncName, IT, double, SSS, SSS);\
    _WRAP_INTEGRATOR_FOR_SPECS(FuncName, IT, double, V1S, V1S);\
    _WRAP_INTEGRATOR_FOR_SPECS(FuncName, IT, double, V2S, V2S);\
    _WRAP_INTEGRATOR_FOR_SPECS(FuncName, IT, double, V3S, V3S);\
    _WRAP_INTEGRATOR_FOR_SPECS(FuncName, IT, double, V4S, V4S);\
    _WRAP_INTEGRATOR_FOR_SPECS(FuncName, IT, double, V5S, V5S);\
    _WRAP_INTEGRATOR_FOR_SPECS(FuncName, IT, double, V6S, V6S);\
    _WRAP_INTEGRATOR_FOR_SPECS(FuncName, IT, double, V7S, V7S);\
    _WRAP_INTEGRATOR_FOR_SPECS(FuncName, IT, double, V8S, V8S);\
    _WRAP_INTEGRATOR_FOR_SPECS(FuncName, IT, double, V9S, V9S);\
    _WRAP_INTEGRATOR_FOR_SPECS(FuncName, IT, double, V10S, V10S);\
    _WRAP_INTEGRATOR_FOR_SPECS(FuncName, IT, double, SO3S, V3S);\
    _WRAP_INTEGRATOR_FOR_SPECS(FuncName, IT, double, SE3S, V6S);\
}

#define WRAP_DYNAMICS_TYPE(DynamicsName, DT, IST) {\
    py::class_<DT>(m, DynamicsName)\
    .def_readwrite("x", &DT::x)\
    .def_readwrite("xdot", &DT::xdot)\
    .def(py::init())\
    .def("setParams", &DT::setParams)\
    .def("hasParams", &DT::hasParams)\
    .def("reset", &DT::reset)\
    .def("t", &DT::t)\
    .def("simulateEuler", static_cast<bool (DT::*)(const IST&, const double&, const bool&, const bool&)>(&DT::simulate<EulerIntegrator>), "Simulate over the whole interval up to t", py::arg("u"), py::arg("tf"), py::arg("insertIntoHistory") = false, py::arg("calculateXddot") = false)\
    .def("simulateEuler", static_cast<bool (DT::*)(const IST&, const double&, const double&, const bool&, const bool&)>(&DT::simulate<EulerIntegrator>), "Simulate over the whole interval up to t in increments of dt", py::arg("u"), py::arg("tf"), py::arg("dt"), py::arg("insertIntoHistory") = false, py::arg("calculateXddot") = false)\
    .def("simulateTrapezoidal", static_cast<bool (DT::*)(const IST&, const double&, const bool&, const bool&)>(&DT::simulate<TrapezoidalIntegrator>), "Simulate over the whole interval up to t", py::arg("u"), py::arg("tf"), py::arg("insertIntoHistory") = false, py::arg("calculateXddot") = false)\
    .def("simulateTrapezoidal", static_cast<bool (DT::*)(const IST&, const double&, const double&, const bool&, const bool&)>(&DT::simulate<TrapezoidalIntegrator>), "Simulate over the whole interval up to t in increments of dt", py::arg("u"), py::arg("tf"), py::arg("dt"), py::arg("insertIntoHistory") = false, py::arg("calculateXddot") = false)\
    .def("simulateSimpson", static_cast<bool (DT::*)(const IST&, const double&, const bool&, const bool&)>(&DT::simulate<SimpsonIntegrator>), "Simulate over the whole interval up to t", py::arg("u"), py::arg("tf"), py::arg("insertIntoHistory") = false, py::arg("calculateXddot") = false)\
    .def("simulateSimpson", static_cast<bool (DT::*)(const IST&, const double&, const double&, const bool&, const bool&)>(&DT::simulate<SimpsonIntegrator>), "Simulate over the whole interval up to t in increments of dt", py::arg("u"), py::arg("tf"), py::arg("dt"), py::arg("insertIntoHistory") = false, py::arg("calculateXddot") = false);\
}

PYBIND11_MODULE(pysignals, m)
{
  m.doc() = "Python binding module for the signals-cpp library.";

  py::enum_<InterpolationMethod>(m, "Interpolation")
    .value("ZERO_ORDER_HOLD", InterpolationMethod::ZERO_ORDER_HOLD)
    .value("LINEAR", InterpolationMethod::LINEAR)
    .value("CUBIC_SPLINE", InterpolationMethod::CUBIC_SPLINE)
    .export_values();

  py::enum_<ExtrapolationMethod>(m, "Extrapolation")
    .value("NANS", ExtrapolationMethod::NANS)
    .value("ZEROS", ExtrapolationMethod::ZEROS)
    .value("CLOSEST", ExtrapolationMethod::CLOSEST)
    .export_values();
  
  py::enum_<DerivativeMethod>(m, "Derivative")
    .value("DIRTY", DerivativeMethod::DIRTY)
    .value("FINITE_DIFF", DerivativeMethod::FINITE_DIFF)
    .export_values();

  WRAP_SIGNAL_TYPE("ScalarSignal", ScalardSignal, ScalardSignal);
  WRAP_SIGNAL_TYPE("Vector1Signal", Vector1dSignal, Vector1dSignal);
  WRAP_SIGNAL_TYPE("Vector2Signal", Vector2dSignal, Vector2dSignal);
  WRAP_SIGNAL_TYPE("Vector3Signal", Vector3dSignal, Vector3dSignal);
  WRAP_SIGNAL_TYPE("Vector4Signal", Vector4dSignal, Vector4dSignal);
  WRAP_SIGNAL_TYPE("Vector5Signal", Vector5dSignal, Vector5dSignal);
  WRAP_SIGNAL_TYPE("Vector6Signal", Vector6dSignal, Vector6dSignal);
  WRAP_SIGNAL_TYPE("Vector7Signal", Vector7dSignal, Vector7dSignal);
  WRAP_SIGNAL_TYPE("Vector8Signal", Vector8dSignal, Vector8dSignal);
  WRAP_SIGNAL_TYPE("Vector9Signal", Vector9dSignal, Vector9dSignal);
  WRAP_SIGNAL_TYPE("Vector10Signal", Vector10dSignal, Vector10dSignal);
  WRAP_SIGNAL_TYPE("SO2Signal", SO2dSignal, Vector1dSignal);
  WRAP_SIGNAL_TYPE("SO3Signal", SO3dSignal, Vector3dSignal);
  WRAP_SIGNAL_TYPE("SE2Signal", SE2dSignal, Vector3dSignal);
  WRAP_SIGNAL_TYPE("SE3Signal", SE3dSignal, Vector6dSignal);

  WRAP_STATE_TYPE("ScalarState", ScalardState, ScalardState);
  WRAP_STATE_TYPE("Vector1State", Vector1dState, Vector1dState);
  WRAP_STATE_TYPE("Vector2State", Vector2dState, Vector2dState);
  WRAP_STATE_TYPE("Vector3State", Vector3dState, Vector3dState);
  WRAP_STATE_TYPE("Vector4State", Vector4dState, Vector4dState);
  WRAP_STATE_TYPE("Vector5State", Vector5dState, Vector5dState);
  WRAP_STATE_TYPE("Vector6State", Vector6dState, Vector6dState);
  WRAP_STATE_TYPE("Vector7State", Vector7dState, Vector7dState);
  WRAP_STATE_TYPE("Vector8State", Vector8dState, Vector8dState);
  WRAP_STATE_TYPE("Vector9State", Vector9dState, Vector9dState);
  WRAP_STATE_TYPE("Vector10State", Vector10dState, Vector10dState);
  WRAP_STATE_TYPE("SO2State", SO2dState, Vector1dState);
  WRAP_STATE_TYPE("SO3State", SO3dState, Vector3dState);
  WRAP_STATE_TYPE("SE2State", SE2dState, Vector3dState);
  WRAP_STATE_TYPE("SE3State", SE3dState, Vector6dState);

  WRAP_SIGNAL_TYPE("ScalarStateSignal", ScalardStateSignal, ScalardStateSignal);
  WRAP_SIGNAL_TYPE("Vector1StateSignal", Vector1dStateSignal, Vector1dStateSignal);
  WRAP_SIGNAL_TYPE("Vector2StateSignal", Vector2dStateSignal, Vector2dStateSignal);
  WRAP_SIGNAL_TYPE("Vector3StateSignal", Vector3dStateSignal, Vector3dStateSignal);
  WRAP_SIGNAL_TYPE("Vector4StateSignal", Vector4dStateSignal, Vector4dStateSignal);
  WRAP_SIGNAL_TYPE("Vector5StateSignal", Vector5dStateSignal, Vector5dStateSignal);
  WRAP_SIGNAL_TYPE("Vector6StateSignal", Vector6dStateSignal, Vector6dStateSignal);
  WRAP_SIGNAL_TYPE("Vector7StateSignal", Vector7dStateSignal, Vector7dStateSignal);
  WRAP_SIGNAL_TYPE("Vector8StateSignal", Vector8dStateSignal, Vector8dStateSignal);
  WRAP_SIGNAL_TYPE("Vector9StateSignal", Vector9dStateSignal, Vector9dStateSignal);
  WRAP_SIGNAL_TYPE("Vector10StateSignal", Vector10dStateSignal, Vector10dStateSignal);
  WRAP_SIGNAL_TYPE("SO2StateSignal", SO2dStateSignal, Vector1dStateSignal);
  WRAP_SIGNAL_TYPE("SO3StateSignal", SO3dStateSignal, Vector3dStateSignal);
  WRAP_SIGNAL_TYPE("SE2StateSignal", SE2dStateSignal, Vector3dStateSignal);
  WRAP_SIGNAL_TYPE("SE3StateSignal", SE3dStateSignal, Vector6dStateSignal);

  WRAP_INTEGRATOR_TYPE("integrateEuler", EulerIntegrator);
  WRAP_INTEGRATOR_TYPE("integrateTrapezoidal", TrapezoidalIntegrator);
  WRAP_INTEGRATOR_TYPE("integrateSimpson", SimpsonIntegrator);

  py::class_<RigidBodyParams1D>(m, "RigidBodyParams1D")
    .def(py::init())
    .def_readwrite("m", &RigidBodyParams1D::m)
    .def_readwrite("g", &RigidBodyParams1D::g);
    
  py::class_<RigidBodyParams2D>(m, "RigidBodyParams2D")
    .def(py::init())
    .def_readwrite("m", &RigidBodyParams2D::m)
    .def_readwrite("J", &RigidBodyParams2D::J)
    .def_readwrite("g", &RigidBodyParams2D::g);

  py::class_<RigidBodyParams3D>(m, "RigidBodyParams3D")
    .def(py::init())
    .def_readwrite("m", &RigidBodyParams3D::m)
    .def_readwrite("J", &RigidBodyParams3D::J)
    .def_readwrite("g", &RigidBodyParams3D::g); 
  
  WRAP_DYNAMICS_TYPE("Translational1DOFModel", Translational1DOFModeld, ScalardSignal);
  WRAP_DYNAMICS_TYPE("Translational2DOFModel", Translational2DOFModeld, Vector2dSignal);
  WRAP_DYNAMICS_TYPE("Translational3DOFModel", Translational3DOFModeld, Vector3dSignal);
  WRAP_DYNAMICS_TYPE("Rotational1DOFModel", Rotational1DOFModeld, Vector1dSignal);
  WRAP_DYNAMICS_TYPE("Rotational3DOFModel", Rotational3DOFModeld, Vector3dSignal);
  WRAP_DYNAMICS_TYPE("RigidBody3DOFModel", RigidBody3DOFModeld, Vector3dSignal);
  WRAP_DYNAMICS_TYPE("RigidBody6DOFModel", RigidBody6DOFModeld, Vector6dSignal);
}
