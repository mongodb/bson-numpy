import glob
import os
import sys

# Suppress warnings during shutdown, http://bugs.python.org/issue15881
try:
    import multiprocessing
except ImportError:
    pass

import setuptools
from setuptools.command.build_ext import build_ext as _build_ext


with open('README.rst') as f:
    readme_content = f.read()

# See http://stackoverflow.com/questions/19919905, we need to install NumPy
# during setup before building bsonnumpy.so, then use its get_include().
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        try:
            # Prevent numpy from thinking it is still in its setup process:
            __builtins__.__NUMPY_SETUP__ = False
        except Exception as exc:
            print("Warning: %s" % exc)
        import numpy
        self.include_dirs.append(numpy.get_include())


libraries = []
if sys.platform == "win32":
    libraries.append("ws2_32")
elif sys.platform != "darwin":
    # librt may be needed for clock_gettime()
    libraries.append("rt")


bsonnumpymodule = setuptools.Extension(
    'bsonnumpy',
    sources=glob.glob("bson-numpy/*.c") + glob.glob("bson-numpy/*/*.c"),
    include_dirs=["bson-numpy", "bson-numpy/bson"],
    define_macros=[("BSON_COMPILATION", 1)],
    libraries=libraries)


setup_requires = ["numpy>=1.17.0"]
tests_require = ["pymongo>=3.9.0"]
test_suite = "test"


# For developing BSON-NumPy with CLion and CMake, useful to disable setuptools.
ext_modules = []
if 'BSONNUMPY_DISABLE_BUILD' not in os.environ:
    ext_modules.append(bsonnumpymodule)

setuptools.setup(
    name='BSON-NumPy',
    version='0.2.0.dev0',
    description='Module for converting directly from BSON to NumPy ndarrays',
    long_description=readme_content,
    author='Anna Herlihy',
    author_email='mongodb-user@googlegroups.com',
    url='https://github.com/mongodb/bson-numpy',
    keywords=["mongo", "mongodb", "pymongo", "numpy", "bson"],
    license="Apache License, Version 2.0",
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Database"],
    ext_modules=ext_modules,
    test_suite=test_suite,
    tests_require=tests_require,
    setup_requires=setup_requires,
    cmdclass={'build_ext': build_ext},
)
