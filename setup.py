import os
import sys

import setuptools
from setuptools.command.build_ext import build_ext as _build_ext

bson_src = os.getenv('BSON_DIR', os.path.join("/usr", "local"))
libraries = ["bson-1.0"]


# See http://stackoverflow.com/questions/19919905, we need to install NumPy
# during setup before building bsonnumpy.so, then use its get_include().
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


bsonnumpymodule = setuptools.Extension(
    'bsonnumpy',
    define_macros=[('MAJOR_VERSION', '0'), ('MINOR_VERSION', '1')],
    include_dirs=[os.path.join(bson_src, "include", "libbson-1.0"),
                  "/usr/include/libbson-1.0"],
    library_dirs=[os.path.join(bson_src, "lib")],
    libraries=libraries,
    extra_compile_args=['-g', '-O0', '-std=c99'],
    extra_link_args=['-g', '-O0'],
    sources=[os.path.join("bson-numpy", "bsonnumpy.c")])


if sys.version_info[:2] == (2, 6):
    # NumPy 1.12 dropped Python 2.6. NumPy 1.11 requires nose even to install.
    setup_requires = ["numpy>=1.11,<1.12", "nose"]
    tests_require = ["unittest2", "nose"]
    test_suite = "unittest2.collector"
else:
    setup_requires = ["numpy"]
    tests_require = []
    test_suite = "test"


setuptools.setup(
    name='BSON-NumPy',
    version='0.1',
    description='Module for converting directly from BSON to NumPy ndarrays'
                ' and vice versa',
    author='Anna Herlihy',
    author_email='anna@mongodb',
    url='https://github.com/aherlihy/numpy-bson',
    ext_modules=[bsonnumpymodule],
    test_suite=test_suite,
    tests_require=tests_require,
    setup_requires=setup_requires,
    install_requires=['pymongo'],
    cmdclass={'build_ext': build_ext},
)
