cmake_minimum_required(VERSION 3.3)

include(FindPythonInterp)
message(STATUS "Found Python ${PYTHON_VERSION_STRING} ${PYTHON_EXECUTABLE}")
include(FindPythonLibs)
include(cmake/FindNumpy.cmake)
include(cmake/FindBSON.cmake)
include_directories(BEFORE ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR} ${BSON_INCLUDE_DIR})

add_library(bsonnumpy SHARED bson-numpy/bsonnumpy.c)
target_link_libraries(bsonnumpy ${PYTHON_LIBRARIES} ${BSON_LIBRARIES})
target_compile_options(bsonnumpy PUBLIC -g)

if (APPLE)
    set_property(TARGET bsonnumpy PROPERTY PREFIX "")
    set_property(TARGET bsonnumpy PROPERTY SUFFIX "")
    set_property(TARGET bsonnumpy PROPERTY OUTPUT_NAME "bsonnumpy.so")
endif()

