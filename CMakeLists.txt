cmake_minimum_required(VERSION 3.3)

set(Python_ADDITIONAL_VERSIONS 2.7)
include(FindPythonInterp)
include(FindPythonLibs)
include(cmake/FindNumpy.cmake)
include(cmake/FindBSON.cmake)
include_directories(BEFORE ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR} ${BSON_INCLUDE_DIR})

add_executable(bsonnumpy bson-numpy/bsonnumpy.c)
target_link_libraries(bsonnumpy ${PYTHON_LIBRARIES} ${BSON_LIBRARIES})
