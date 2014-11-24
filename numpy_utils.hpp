#include "fcl/math/transform.h"

namespace py = boost::python;
py::object np_mod; 

template<typename T>
struct type_traits {
  static const char* npname;
};
template<> const char* type_traits<float>::npname = "float32";
template<> const char* type_traits<int>::npname = "int32";
template<> const char* type_traits<double>::npname = "float64";
template<> const char* type_traits<unsigned char>::npname = "uint8";

template <typename T>
T* getPointer(const py::object& arr) {
  long int i = py::extract<long int>(arr.attr("ctypes").attr("data"));
  T* p = (T*)i;
  return p;
}

template<typename T>
py::object toNdarray1(const T* data, size_t dim0) {
  py::object out = np_mod.attr("empty")(py::make_tuple(dim0), type_traits<T>::npname);
  T* p = getPointer<T>(out);
  memcpy(p, data, dim0*sizeof(T));
  return out;
}
template<typename T>
py::object toNdarray2(const T* data, size_t dim0, size_t dim1) {
  py::object out = np_mod.attr("empty")(py::make_tuple(dim0, dim1), type_traits<T>::npname);
  T* pout = getPointer<T>(out);
  memcpy(pout, data, dim0*dim1*sizeof(T));
  return out;
}
template<typename T>
py::object toNdarray3(const T* data, size_t dim0, size_t dim1, size_t dim2) {
  py::object out = np_mod.attr("empty")(py::make_tuple(dim0, dim1, dim2), type_traits<T>::npname);
  T* pout = getPointer<T>(out);
  memcpy(pout, data, dim0*dim1*dim2*sizeof(T));
  return out;
}

py::object Transform3fToNdarray2(const fcl::Transform3f& transform) {
  size_t dim0 = 4, dim1 = 4;
  py::object out = np_mod.attr("empty")(py::make_tuple(dim0, dim1), type_traits<double>::npname);
  double* pout = getPointer<double>(out);
  memset(pout, 0, dim0*dim1*sizeof(double));
  memcpy(pout, (double*)(transform.getRotation().data.rs[0].vs), 3*sizeof(double));
  memcpy(pout+3, (double*)(transform.getTranslation().data.vs), 1*sizeof(double));
  memcpy(pout+4, (double*)(transform.getRotation().data.rs[1].vs), 3*sizeof(double));
  memcpy(pout+7, (double*)(transform.getTranslation().data.vs+1), 1*sizeof(double));
  memcpy(pout+8, (double*)(transform.getRotation().data.rs[2].vs), 3*sizeof(double));
  memcpy(pout+11, (double*)(transform.getTranslation().data.vs+2), 1*sizeof(double));
  std::fill_n(pout+12, 0.0, 3);
  std::fill_n(pout+15, 1.0, 1);
  return out;
}