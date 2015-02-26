#include <boost/python.hpp>
#include <boost/foreach.hpp>
#include <boost/shared_array.hpp>
#include "fcl/shape/geometric_shapes.h"
#include "fcl/math/vec_3f.h"
#include "fcl/math/matrix_3f.h"
#include "fcl/math/transform.h"
#include "fcl/collision_object.h"
#include "fcl/collision_data.h"
#include "fcl/collision.h"
#include "fcl/distance.h"
#include "fcl/broadphase/broadphase.h"
#include "fcl/ccd/motion.h"
#include "numpy_utils.hpp"

#include <fenv.h>

using namespace fcl;

namespace py = boost::python;

template<typename T, typename PyT>
py::list toPyList(const std::vector<T>& in) {
  py::list out;
  BOOST_FOREACH(const T& elem, in) {
    out.append(PyT(elem));
  }
  return out;
}

struct ObjectInfo {
  std::string _name;
  py::numeric::array _t;
  py::numeric::array _t_inv;
  ObjectInfo(py::numeric::array t) : _t((py::numeric::array)t.attr("copy")()), 
                                     _t_inv((py::numeric::array)np_mod.attr("linalg").attr("inv")(t))
                                     {}
};

void copyTransformPytoCpp(Transform3f* cpp_t, py::numeric::array& py_t) {
  Vec3f R_row_1 = Vec3f(py::extract<double>(py_t[0][0]),py::extract<double>(py_t[0][1]),py::extract<double>(py_t[0][2]));
  R_row_1 = R_row_1.normalize();
  Vec3f R_row_2 = Vec3f(py::extract<double>(py_t[1][0]),py::extract<double>(py_t[1][1]),py::extract<double>(py_t[1][2]));
  R_row_2 = R_row_2.normalize();
  Vec3f R_row_3 = Vec3f(py::extract<double>(py_t[2][0]),py::extract<double>(py_t[2][1]),py::extract<double>(py_t[2][2]));
  R_row_3 = R_row_3.normalize();
  Matrix3f R = Matrix3f(R_row_1, R_row_2, R_row_3);
  Vec3f T = Vec3f(py::extract<double>(py_t[0][3]),py::extract<double>(py_t[1][3]),py::extract<double>(py_t[2][3]));
  *cpp_t = Transform3f(R,T);
}

/// @brief Collision data stores the collision request and the result given by collision algorithm. 
struct CollisionData
{
  CollisionData()
  {
    done = false;
  }

  /// @brief Collision request
  CollisionRequest request;

  /// @brief Collision result
  CollisionResult result;

  /// @brief Whether the collision iteration can stop
  bool done;
};

/// @brief Distance data stores the distance request and the result given by distance algorithm. 
struct DistanceData
{
  DistanceData()
  {
    done = false;
  }

  /// @brief Distance request
  DistanceRequest request;

  /// @brief Distance result
  DistanceResult result;

  /// @brief Whether the distance iteration can stop
  bool done;
};

bool defaultCollisionFunction(CollisionObject* o1, CollisionObject* o2, void* cdata_)
{
  CollisionData* cdata = static_cast<CollisionData*>(cdata_);
  const CollisionRequest& request = cdata->request;
  CollisionResult& result = cdata->result;

  if(cdata->done) return true;

  collide(o1, o2, request, result);

  if(!request.enable_cost && (result.isCollision()) && (result.numContacts() >= request.num_max_contacts))
    cdata->done = true;

  return cdata->done;
}

bool defaultDistanceFunction(CollisionObject* o1, CollisionObject* o2, void* cdata_, FCL_REAL& dist)
{
  DistanceData* cdata = static_cast<DistanceData*>(cdata_);
  const DistanceRequest& request = cdata->request;
  DistanceResult& result = cdata->result;

  if(cdata->done) { dist = result.min_distance; return true; }

  distance(o1, o2, request, result);
  dist = result.min_distance;

  if(dist <= 0) return true; // in collision or in touch

  return cdata->done;
}

//New class to store name
class CollisionObjectRLL : public CollisionObject {
  ObjectInfo obj_info;
public:
  CollisionObjectRLL(boost::shared_ptr<CollisionGeometry> geom, py::numeric::array& py_t_offset): CollisionObject(geom),
                                                                      obj_info(py_t_offset)
                                                                      { Transform3f temp_t;
                                                                        copyTransformPytoCpp(&temp_t,obj_info._t);
                                                                        this->setTransform(temp_t);

                                                                        //Hack to make sure ObjectInfo can be recovered from Contact objects
                                                                        user_data = (void*)&obj_info;
                                                                        cgeom.get()->setUserData(user_data);
                                                                      }
  void setName(char* name) {
    obj_info._name = std::string(name);
  }
  const char* getName() const{
    return obj_info._name.c_str();
  }
  const py::numeric::array& getTransformOffset() const{
    return obj_info._t;
  }
  const py::numeric::array& getInvTransformOffset() const{
    return obj_info._t_inv;
  }

};

//Collision objects
class PyCollisionObject {
public:
  boost::shared_ptr<CollisionObjectRLL> m_obj;
  PyCollisionObject(boost::shared_ptr<CollisionObjectRLL> obj) : m_obj(obj) {}
  NODE_TYPE GetNodeType() {
    return m_obj->getNodeType();
  }
  void SetTransform(py::numeric::array& py_t) {
    py_t = (py::numeric::array)py_t.attr("dot")(m_obj->getTransformOffset());
    Transform3f cpp_t;
    copyTransformPytoCpp(&cpp_t,py_t);
    m_obj->setTransform(cpp_t);
  }
  py::numeric::array GetTransform() {return (py::numeric::array)Transform3fToNdarray2(m_obj->getTransform()).attr("dot")(m_obj->getInvTransformOffset());}
  void SetName(py::str py_name) {m_obj->setName(py::extract<char*>(py_name));}
  py::str GetName() {return py::str(m_obj->getName());}
};

class PyCollisionBox: public PyCollisionObject {
public:
  PyCollisionBox(boost::shared_ptr<CollisionObjectRLL> box): PyCollisionObject(box) {}
  double GetX() {
    return ((Box*)(m_obj->collisionGeometry().get()))->side[0];
  }
  double GetY() {
    return ((Box*)(m_obj->collisionGeometry().get()))->side[1];
  }
  double GetZ() {
    return ((Box*)(m_obj->collisionGeometry().get()))->side[2];
  }
};

class PyCollisionCylinder: public PyCollisionObject {
public:
  PyCollisionCylinder(boost::shared_ptr<CollisionObjectRLL> cyl): PyCollisionObject(cyl) {}
  double GetRadius() {
    return ((Cylinder*)(m_obj->collisionGeometry().get()))->radius;
  }
  double GetHeight() {
    return ((Cylinder*)(m_obj->collisionGeometry().get()))->lz;
  }
};

class PyCollisionSphere: public PyCollisionObject {
public:
  PyCollisionSphere(boost::shared_ptr<CollisionObjectRLL> sph): PyCollisionObject(sph) {}
  double GetRadius() {
    return ((Sphere*)(m_obj->collisionGeometry().get()))->radius;
  }
};

class PyCollisionMesh: public PyCollisionObject {
public:
  PyCollisionMesh(boost::shared_ptr<CollisionObjectRLL> mesh): PyCollisionObject(mesh) {}
  py::object GetPoints() {
    const std::vector<Vec3f>* cpp_points = &(((Convex*)(m_obj->collisionGeometry().get()))->vec_points);
    py::list py_points;
    BOOST_FOREACH(const Vec3f& cpp_point, *cpp_points) {
      py::list py_point;
      py_point.append(cpp_point[0]);
      py_point.append(cpp_point[1]);
      py_point.append(cpp_point[2]);
      py_points.append(py_point);
    }
    return py_points;
  }
  py::object GetSimplices() {
    const std::vector<int>* indicies = &(((Convex*)(m_obj->collisionGeometry().get()))->vec_polygons);
    py::list py_simplicies;
    for (size_t i=0; i < indicies->size(); i++){
      size_t num_points = (*indicies)[i];
      py::list py_simplex;
      for (size_t k=0; k < num_points; k++){
        py_simplex.append((*indicies)[++i]);
      }
      py_simplicies.append(py_simplex);
    }
    return py_simplicies;
  }
};

PyCollisionBox createBox(py::object& py_dims, py::numeric::array py_t_offset=(py::numeric::array)np_mod.attr("eye")(4)) {
  Box* box = new Box(py::extract<double>(py_dims[0]),py::extract<double>(py_dims[1]),py::extract<double>(py_dims[2]));
  CollisionObjectRLL* obj = new CollisionObjectRLL(boost::shared_ptr<CollisionGeometry>(box), py_t_offset);
  return PyCollisionBox(boost::shared_ptr<CollisionObjectRLL>(obj));
}

PyCollisionCylinder createCylinder(py::object& py_dims, py::numeric::array py_t_offset=(py::numeric::array)np_mod.attr("eye")(4)) {
  Cylinder* cyl = new Cylinder(py::extract<double>(py_dims[0]),py::extract<double>(py_dims[1]));
  CollisionObjectRLL* obj = new CollisionObjectRLL(boost::shared_ptr<CollisionGeometry>(cyl), py_t_offset);
  return PyCollisionCylinder(boost::shared_ptr<CollisionObjectRLL>(obj));
}

PyCollisionSphere createSphere(double py_r, py::numeric::array py_t_offset=(py::numeric::array)np_mod.attr("eye")(4)) {
  Sphere* sph = new Sphere(py_r);
  CollisionObjectRLL* obj = new CollisionObjectRLL(boost::shared_ptr<CollisionGeometry>(sph), py_t_offset);
  return PyCollisionSphere(boost::shared_ptr<CollisionObjectRLL>(obj));
}

PyCollisionMesh createTriMesh(py::object& py_points, py::object& py_indices, py::numeric::array py_t_offset=(py::numeric::array)np_mod.attr("eye")(4)){
  const int verticies_per_plane = 3;
  int num_points = boost::python::extract<int>(py_points.attr("__len__")());
  std::vector<Vec3f> cpp_points;

  for (int i=0; i < num_points; i++) {
    cpp_points.push_back(Vec3f(py::extract<double>(py_points[i][0]),py::extract<double>(py_points[i][1]),py::extract<double>(py_points[i][2])));
  }

  int num_planes = boost::python::extract<int>(py_indices.attr("__len__")());
  std::vector<Vec3f> plane_normals;
  std::vector<int> polygons;

  //TODO: Sort the points in counter clockwise order before computing the normals!
  for (int i=0; i < num_planes; i++) {
    int index = i*(verticies_per_plane+1);
    polygons.push_back(verticies_per_plane);
    index++;
    for (int j=0; j < verticies_per_plane; j++) {
      polygons.push_back(py::extract<int>(py_indices[i][j]));
    }
    Vec3f edge1 = cpp_points[polygons[index+1]]-cpp_points[polygons[index]];
    Vec3f edge2 = cpp_points[polygons[index+2]]-cpp_points[polygons[index]];
    plane_normals.push_back(edge1.cross(edge2));
  }

  Convex* mesh = new Convex(plane_normals,num_planes,cpp_points,num_points,polygons);
  CollisionObjectRLL* obj = new CollisionObjectRLL(boost::shared_ptr<CollisionGeometry>(mesh), py_t_offset);
  return PyCollisionMesh(boost::shared_ptr<CollisionObjectRLL>(obj));
}


//Continuous collision objects
/*class PyContinuousCollisionObject {
public:
  ContinuousCollisionObject* m_obj;
  PyContinuousCollisionObject(ContinuousCollisionObject* obj) : m_obj(obj) {}

  NODE_TYPE GetNodeType() {
    return m_obj->getNodeType();
  }
  py::object GetTransform(double dt=0.0){
    MotionBase* motion = m_obj->getMotion();
    motion->integrate(dt);
    Transform3f curr_t;
    motion->getCurrentTransform(curr_t);
    return Transform3fToNdarray2(curr_t);
  }
  py::str GetName() {return py::str((const char*)(m_obj->getUserData()));}
};

PyContinuousCollisionObject createBoxInMotion(py::object py_dims, py::object py_t1, py::object py_t2, py::str py_name){
  Transform3f* start_t = toFclTransform(py_t1);
  Transform3f* end_t = toFclTransform(py_t2);
  char* name = toCppString(py_name);

  Box* box = new Box(py::extract<double>(py_dims[0]),py::extract<double>(py_dims[1]),py::extract<double>(py_dims[2]));
  box->setUserData((void*)name);
  MotionBase* motion = new InterpMotion(*start_t,*end_t);
  ContinuousCollisionObject* obj = new ContinuousCollisionObject(boost::shared_ptr<CollisionGeometry>(box), boost::shared_ptr<MotionBase>(motion));
  obj->setUserData((void*)name);

  return PyContinuousCollisionObject(obj);
}*/

class PyContact {
public:
  Contact m_c;
  PyContact(const Contact& c) : m_c(c) {}

  double GetPenDepth() {return m_c.penetration_depth;}
  py::object GetNormal() {return toNdarray1<double>((double*)m_c.normal.data.vs,3);}
  py::object GetPtA() {return toNdarray1<double>((double*)m_c.pos.data.vs,3);}
  py::object GetPtB() {
    Vec3f PtB = m_c.pos+m_c.normal*m_c.penetration_depth;
    return toNdarray1<double>((double*)&PtB.data.vs,3);
  }
  py::str GetGeomAName() {return py::str(((const ObjectInfo*)(m_c.o1->getUserData()))->_name);}
  py::str GetGeomBName() {return py::str(((const ObjectInfo*)(m_c.o2->getUserData()))->_name);}

    // return (char*)user_data;
};

bool compareContacts(const Contact& c1,const Contact& c2) { return c1.penetration_depth > c2.penetration_depth; } //Decending order (Assuming penetration depth is a positive number)!!

class PyDynamicAABBTreeCollisionManager {
public:
  PyDynamicAABBTreeCollisionManager(boost::shared_ptr<DynamicAABBTreeCollisionManager> cm) : m_cm(cm) {}
  void RegisterObjects(const py::list py_objs) {
    int n_objs = boost::python::extract<int>(py_objs.attr("__len__")());
    if (n_objs > 0) {
      std::vector<CollisionObject*> cpp_objs(n_objs);
      for (int i=0; i < n_objs; ++i) {
        PyCollisionObject* cpp_obj = py::extract<PyCollisionObject*>(py_objs[i]);
        cpp_objs[i] = cpp_obj->m_obj.get();
        m_objs.push_back(cpp_obj->m_obj);
      }
      m_cm->registerObjects(cpp_objs);
    }
  }
  void RegisterObject(py::object py_obj){
    PyCollisionObject* cpp_obj = py::extract<PyCollisionObject*>(py_obj);
    m_cm->registerObject(cpp_obj->m_obj.get());
    m_objs.push_back(cpp_obj->m_obj);
  }
  void UnregisterObjects(const py::list py_objs) {
    int n_objs = boost::python::extract<int>(py_objs.attr("__len__")());
    for (int i=0; i < n_objs; ++i) {
      PyCollisionObject* cpp_obj = py::extract<PyCollisionObject*>(py_objs[i]);
      m_cm->unregisterObject(cpp_obj->m_obj.get());
      m_objs.erase(std::remove(m_objs.begin(), m_objs.end(), cpp_obj->m_obj), m_objs.end());
    }
  }
  void UnregisterObject(py::object py_obj) {
    PyCollisionObject* cpp_obj = py::extract<PyCollisionObject*>(py_obj);
    m_cm->unregisterObject(cpp_obj->m_obj.get());
    m_objs.erase(std::remove(m_objs.begin(), m_objs.end(), cpp_obj->m_obj), m_objs.end());
  }
  py::list GetObjects() {
    int n_objs = m_cm->size();
    py::list py_objs;
    if (n_objs > 0) {
      for (int i=0; i < n_objs; ++i) {
        switch (m_objs.at(i)->getNodeType()) {
          case GEOM_BOX:
            py_objs.append(PyCollisionBox(m_objs.at(i)));
            break;
          case GEOM_CYLINDER:
            py_objs.append(PyCollisionCylinder(m_objs.at(i)));
            break;
          case GEOM_CONVEX:
            py_objs.append(PyCollisionMesh(m_objs.at(i)));
            break;
          default:
            std::cout << "Unknown geometry type!" << std::endl;
        }  
      }
    }
    return py_objs;
  }

  py::object BodyVsAllCollide(py::object py_obj, bool sort=true){
    PyCollisionObject* cpp_obj = py::extract<PyCollisionObject*>(py_obj);
    CollisionData cdata;
    cdata.request.num_max_contacts = 100000;
    cdata.request.enable_contact = true;
    m_cm->collide(cpp_obj->m_obj.get(), &cdata, defaultCollisionFunction);
    std::vector<Contact> contacts;
    cdata.result.getContacts(contacts);
    if (cdata.result.numContacts()>0){
      std::sort(contacts.begin(), contacts.end(), compareContacts);
    }
    return toPyList<Contact,PyContact>(contacts);
  }

  py::object CmVsCmCollide(py::object py_obj, bool sort=true){
    PyDynamicAABBTreeCollisionManager* cpp_cm = py::extract<PyDynamicAABBTreeCollisionManager*>(py_obj);
    CollisionData cdata;
    cdata.request.num_max_contacts = 100000;
    cdata.request.enable_contact = true;
    m_cm->collide(cpp_cm->m_cm.get(), &cdata, defaultCollisionFunction);
    std::vector<Contact> contacts;
    cdata.result.getContacts(contacts);
    if (cdata.result.numContacts()>0){
      std::sort(contacts.begin(), contacts.end(), compareContacts);
      //std::cout << contacts.front().pos << "\n";
    }
    return toPyList<Contact,PyContact>(contacts);
  }

  /// @brief perform collision test for the objects belonging to the manager (i.e., N^2 self collision)
  py::object Collide(bool sort=true){
    CollisionData cdata;
    cdata.request.num_max_contacts = 100000;
    cdata.request.enable_contact = true;
    m_cm->collide(&cdata, defaultCollisionFunction);
    std::vector<Contact> contacts;
    cdata.result.getContacts(contacts);
    if (cdata.result.numContacts()>0){
      std::sort(contacts.begin(), contacts.end(), compareContacts);
    }
    return toPyList<Contact,PyContact>(contacts);
  }

  double BodyVsAllDistance(py::object py_obj){
    PyCollisionObject* cpp_obj = py::extract<PyCollisionObject*>(py_obj);
    DistanceData cdata;
    m_cm->distance(cpp_obj->m_obj.get(), &cdata, defaultDistanceFunction);
    return cdata.result.min_distance;
  }
  /// @brief perform distance test for the objects belonging to the manager (i.e., N^2 self distance)
  double Distance(){
    DistanceData cdata;
    //if(exhaustive) cdata.request.num_max_contacts = 100000;
    m_cm->distance(&cdata, defaultDistanceFunction);
    return cdata.result.min_distance;
    //m_cm->distance(void* cdata, DistanceCallBack callback) const;
  }

  void Update() {
    m_cm->update();
  }

  void Clear() {
    m_cm->clear();
  }

  /*
  /// @brief perform collision test between one object and all the objects belonging to the manager
  void collide(CollisionObject* obj, void* cdata, CollisionCallBack callback) const;

  /// @brief perform distance computation between one object and all the objects belonging to the manager
  void distance(CollisionObject* obj, void* cdata, DistanceCallBack callback) const;

  /// @brief perform collision test with objects belonging to another manager
  void collide(BroadPhaseCollisionManager* other_manager_, void* cdata, CollisionCallBack callback) const;

  /// @brief perform distance test with objects belonging to another manager
  void distance(BroadPhaseCollisionManager* other_manager_, void* cdata, DistanceCallBack callback) const;
  */
private:
  boost::shared_ptr<DynamicAABBTreeCollisionManager> m_cm;
  std::vector< boost::shared_ptr<CollisionObjectRLL> > m_objs;
};

PyDynamicAABBTreeCollisionManager createCollisionManager() {
  DynamicAABBTreeCollisionManager* cm = new DynamicAABBTreeCollisionManager();
  return PyDynamicAABBTreeCollisionManager(boost::shared_ptr<DynamicAABBTreeCollisionManager>(cm));
}

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(BodyVsAllCollideDefaults, PyDynamicAABBTreeCollisionManager::BodyVsAllCollide, 1, 2);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(CmVsCmCollideDefaults, PyDynamicAABBTreeCollisionManager::CmVsCmCollide, 1, 2);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(CollideDefaults, PyDynamicAABBTreeCollisionManager::Collide, 0, 1);
BOOST_PYTHON_FUNCTION_OVERLOADS(createBoxDefaults, createBox, 1, 2);
BOOST_PYTHON_FUNCTION_OVERLOADS(createCylDefaults, createCylinder, 1, 2);
BOOST_PYTHON_FUNCTION_OVERLOADS(createSphDefaults, createSphere, 1, 2);
BOOST_PYTHON_FUNCTION_OVERLOADS(createTriDefaults, createTriMesh, 2, 3);

BOOST_PYTHON_MODULE(fclpy) {

  np_mod = py::import("numpy");

  py::class_<PyCollisionObject>("PyCollisionObject", py::no_init)
    .def("GetNodeType", &PyCollisionObject::GetNodeType)
    .def("GetTransform", &PyCollisionObject::GetTransform)
    .def("SetTransform", &PyCollisionObject::SetTransform)
    .def("SetName", &PyCollisionObject::SetName)
    .def("GetName", &PyCollisionObject::GetName)
    ;
  py::class_<PyCollisionBox, py::bases<PyCollisionObject> >("PyCollisionBox", py::no_init)
    .def("GetX", &PyCollisionBox::GetX)
    .def("GetY", &PyCollisionBox::GetY)
    .def("GetZ", &PyCollisionBox::GetZ)
    ;
  py::class_<PyCollisionCylinder, py::bases<PyCollisionObject> >("PyCollisionCylinder", py::no_init)
    .def("GetRadius", &PyCollisionCylinder::GetRadius)
    .def("GetHeight", &PyCollisionCylinder::GetHeight)
    ;
  py::class_<PyCollisionSphere, py::bases<PyCollisionObject> >("PyCollisionSphere", py::no_init)
    .def("GetRadius", &PyCollisionSphere::GetRadius)
    ;
  py::class_<PyCollisionMesh, py::bases<PyCollisionObject> >("PyCollisionMesh", py::no_init)
    .def("GetPoints", &PyCollisionMesh::GetPoints)
    .def("GetSimplices", &PyCollisionMesh::GetSimplices)
    ;

  py::def("createBox", &createBox, createBoxDefaults());
  py::def("createCylinder", &createCylinder, createCylDefaults());
  py::def("createSphere", &createSphere, createSphDefaults());
  py::def("createTriMesh", &createTriMesh, createTriDefaults());

  /*py::class_<PyContinuousCollisionObject>("PyContinuousCollisionObject", py::no_init)
    .def("GetNodeType", &PyContinuousCollisionObject::GetNodeType)
    .def("GetTransform", &PyContinuousCollisionObject::GetTransform, ContGetTransformDefaults())
    .def("GetName", &PyCollisionObject::GetName)
    ;
  py::def("createBoxInMotion", &createBoxInMotion);*/

  py::class_<PyContact>("PyContact", py::no_init)
    .def("GetPenDepth", &PyContact::GetPenDepth)
    .def("GetNormal", &PyContact::GetNormal)
    .def("GetPtA", &PyContact::GetPtA)
    .def("GetPtB", &PyContact::GetPtB)
    .def("GetGeomAName", &PyContact::GetGeomAName)
    .def("GetGeomBName", &PyContact::GetGeomBName)
    ;

  py::class_<PyDynamicAABBTreeCollisionManager>("PyDynamicAABBTreeCollisionManager", py::no_init)
    .def("RegisterObjects", &PyDynamicAABBTreeCollisionManager::RegisterObjects)
    .def("RegisterObject", &PyDynamicAABBTreeCollisionManager::RegisterObject)
    .def("UnregisterObjects", &PyDynamicAABBTreeCollisionManager::UnregisterObjects)
    .def("UnregisterObject", &PyDynamicAABBTreeCollisionManager::UnregisterObject)
    .def("GetObjects", &PyDynamicAABBTreeCollisionManager::GetObjects)
    .def("Collide", &PyDynamicAABBTreeCollisionManager::Collide, CollideDefaults())
    .def("Distance", &PyDynamicAABBTreeCollisionManager::Distance)
    .def("BodyVsAllCollide", &PyDynamicAABBTreeCollisionManager::BodyVsAllCollide, BodyVsAllCollideDefaults())
    .def("CmVsCmCollide", &PyDynamicAABBTreeCollisionManager::CmVsCmCollide, CmVsCmCollideDefaults())
    .def("BodyVsAllDistance", &PyDynamicAABBTreeCollisionManager::BodyVsAllDistance)
    .def("Update", &PyDynamicAABBTreeCollisionManager::Update)
    .def("Clear", &PyDynamicAABBTreeCollisionManager::Clear)
    ;
  py::def("createCollisionManager", &createCollisionManager);

  py::enum_<NODE_TYPE>("NODE_TYPE")
    .value("GEOM_BOX", GEOM_BOX)
    .value("GEOM_SPHERE", GEOM_SPHERE)
    .value("GEOM_CONE", GEOM_CONE)
    .value("GEOM_CYLINDER", GEOM_CYLINDER)
    .value("GEOM_CAPSULE", GEOM_CAPSULE)
    .value("GEOM_CONVEX", GEOM_CONVEX)
    .value("GEOM_PLANE", GEOM_PLANE)
    .value("GEOM_HALFSPACE", GEOM_HALFSPACE)
    .value("GEOM_TRIANGLE", GEOM_TRIANGLE)
    .value("GEOM_OCTREE", GEOM_OCTREE)
    .value("NODE_COUNT", NODE_COUNT)
    .value("BV_UNKNOWN", BV_UNKNOWN)
    .value("BV_AABB", BV_AABB)
    .value("BV_OBB", BV_OBB)
    .value("BV_RSS", BV_RSS)
    .value("BV_kIOS", BV_kIOS)
    .value("BV_OBBRSS", BV_OBBRSS)
    .value("BV_KDOP16", BV_KDOP16)
    .value("BV_KDOP18", BV_KDOP18)
    .value("BV_KDOP24", BV_KDOP24)
  ;

}
