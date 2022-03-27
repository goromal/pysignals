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
    .def("update", static_cast<bool (BST::*)(const std::vector<double>&, const std::vector<BST::BaseType>&, const std::vector<BST::TangentType>&)>(&BST::update), "Update signal and derivative values", py::arg("tHistory"), py::arg("xHistory"), py::arg("xdotHistory"));\
}

#define _WRAP_INTEGRATOR_FOR_SPECS(IT, intObj, BSS, TSS) {\
    intObj\
    .def("__call__", static_cast<bool (IT::*)(Signal<BSS, TSS>&, const Signal<TSS, TSS>&, const double&, const bool&)>(&IT::operator()), "Integrate over the whole interval up to t", py::arg("xInt"), py::arg("x"), py::arg("t"), py::arg("insertIntoHistory") = false)\
    .def("__call__", static_cast<bool (IT::*)(Signal<BSS, TSS>&, const Signal<TSS, TSS>&, const double&, const double&, const bool&)>(&IT::operator()), "Integrate over the whole interval up to t in increments of dt", py::arg("xInt"), py::arg("x"), py::arg("t"), py::arg("dt"), py::arg("insertIntoHistory") = false);\
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
#define SO3S ManifoldSignalSpec<SO3d>
#define SE3S ManifoldSignalSpec<SE3d>

#define WRAP_INTEGRATOR_TYPE(IntegratorName, IT) {\
    py::class_<IT> intObj(m, IntegratorName);\
    intObj.def(py::init());\
    _WRAP_INTEGRATOR_FOR_SPECS(IT, intObj, SSS, SSS);\
    _WRAP_INTEGRATOR_FOR_SPECS(IT, intObj, V1S, V1S);\
    _WRAP_INTEGRATOR_FOR_SPECS(IT, intObj, V2S, V2S);\
    _WRAP_INTEGRATOR_FOR_SPECS(IT, intObj, V3S, V3S);\
    _WRAP_INTEGRATOR_FOR_SPECS(IT, intObj, V4S, V4S);\
    _WRAP_INTEGRATOR_FOR_SPECS(IT, intObj, V5S, V5S);\
    _WRAP_INTEGRATOR_FOR_SPECS(IT, intObj, V6S, V6S);\
    _WRAP_INTEGRATOR_FOR_SPECS(IT, intObj, V7S, V7S);\
    _WRAP_INTEGRATOR_FOR_SPECS(IT, intObj, V8S, V8S);\
    _WRAP_INTEGRATOR_FOR_SPECS(IT, intObj, V9S, V9S);\
    _WRAP_INTEGRATOR_FOR_SPECS(IT, intObj, V10S, V10S);\
    _WRAP_INTEGRATOR_FOR_SPECS(IT, intObj, SO3S, V3S);\
    _WRAP_INTEGRATOR_FOR_SPECS(IT, intObj, SE3S, V6S);\
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
  WRAP_SIGNAL_TYPE("SO3Signal", SO3dSignal, Vector3dSignal);
  WRAP_SIGNAL_TYPE("SE3Signal", SE3dSignal, Vector6dSignal);

  WRAP_INTEGRATOR_TYPE("IntegrateEuler", IntegrateEuler);
  WRAP_INTEGRATOR_TYPE("IntegrateTrapezoidal", IntegrateTrapezoidal);
}
