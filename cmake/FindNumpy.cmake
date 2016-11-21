# Inspired by a CMake file by Anton Deguet
# Run "include(FindPythonInterp)" first.

# Read-Only variables:
#  NUMPY_FOUND
#  NUMPY_INCLUDE_DIR

# PYTHON_EXECUTABLE is set by FindPythonInterp.
if (PYTHON_EXECUTABLE)
    file(
        WRITE ${CMAKE_CURRENT_BINARY_DIR}/determineNumpyPath.py "
try:
    import numpy
    print numpy.get_include()
except Exception as exc:
    print(exc)")

    exec_program(
        "${PYTHON_EXECUTABLE}"
        ARGS "\"${CMAKE_CURRENT_BINARY_DIR}/determineNumpyPath.py\""
        OUTPUT_VARIABLE NUMPY_PATH)
endif (PYTHON_EXECUTABLE)

find_path(
    NUMPY_INCLUDE_DIR numpy/arrayobject.h
    "${NUMPY_PATH}"
    "${PYTHON_INCLUDE_PATH}"
    DOC
    "Directory where the arrayobject.h header file can be found. \
    This file is part of the numpy package")

if (NUMPY_INCLUDE_DIR)
    message(STATUS "Found NumPy includes: ${NUMPY_INCLUDE_DIR}")
    set(NUMPY_FOUND 1 CACHE INTERNAL "NumPy found")
endif ()
