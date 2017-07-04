cmake_minimum_required(VERSION 3.3)

include(FindPythonInterp)
message(STATUS "Found Python ${PYTHON_VERSION_STRING} ${PYTHON_EXECUTABLE}")
include(FindPythonLibs)
include(cmake/FindNumpy.cmake)
include_directories(
        BEFORE
        ${PYTHON_INCLUDE_DIRS}
        ${NUMPY_INCLUDE_DIR}
        bson-numpy/bson
        bson-numpy/jsonsl)

file(GLOB bsonnumpy_SRC "bson-numpy/bson/*.c" "bson-numpy/jsonsl/*.c")
add_library(bsonnumpy SHARED bson-numpy/bsonnumpy.c ${bsonnumpy_SRC})
target_include_directories(bsonnumpy PRIVATE bson-numpy)
target_link_libraries(bsonnumpy ${PYTHON_LIBRARIES} ${BSON_LIBRARIES})
target_compile_definitions(bsonnumpy PRIVATE BSON_COMPILATION)

if (APPLE)
    set_property(TARGET bsonnumpy PROPERTY PREFIX "")
    set_property(TARGET bsonnumpy PROPERTY SUFFIX "")
    set_property(TARGET bsonnumpy PROPERTY OUTPUT_NAME "bsonnumpy.so")
endif()

