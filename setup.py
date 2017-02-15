import os
import sys

# Suppress warnings during shutdown, http://bugs.python.org/issue15881
try:
    import multiprocessing
except ImportError:
    pass

import setuptools
from setuptools.command.build_ext import build_ext as _build_ext


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


bsonnumpymodule = setuptools.Extension(
    'bsonnumpy',
    libraries=["bson-1.0"],
    include_dirs=["/usr/include/libbson-1.0", "/usr/local/include/libbson-1.0"],
    sources=[os.path.join("bson-numpy", "bsonnumpy.c")])


if sys.version_info[:2] == (2, 6):
    # NumPy 1.12 dropped Python 2.6.
    setup_requires = ["numpy==1.11.2"]
    tests_require = ["pymongo", "unittest2"]
    test_suite = "unittest2.collector"
else:
    setup_requires = ["numpy>=1.11.0"]
    tests_require = ["pymongo"]
    test_suite = "test"


setuptools.setup(
    name='BSON-NumPy',
    version='0.1',
    description='Module for converting directly from BSON to NumPy ndarrays'
                ' and vice versa',
    author='Anna Herlihy',
    author_email='anna@mongodb',
    url='https://github.com/aherlihy/bson-numpy',
    ext_modules=[bsonnumpymodule],
    test_suite=test_suite,
    tests_require=tests_require,
    setup_requires=setup_requires,
    cmdclass={'build_ext': build_ext},
)
