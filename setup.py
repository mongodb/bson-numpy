import os
import sys

import numpy as np
import setuptools

# TODO: use correct setuptools so install_requires works

bson_src = os.getenv('BSON_DIR', os.path.join("/usr", "local"))
libraries = ["bson-1.0"]

# TODO: bootstrap numpy installation so setup.py install works.

bsonnumpymodule = setuptools.Extension(
    'bsonnumpy',
    define_macros=[('MAJOR_VERSION', '0'), ('MINOR_VERSION', '1')],
    include_dirs=[os.path.join(bson_src, "include", "libbson-1.0"),
                  np.get_include()],
    install_requires=['pymongo'],
    library_dirs=[os.path.join(bson_src, "lib")],
    libraries=libraries,
    extra_compile_args=['-g', '-O0', '-std=c99'],
    extra_link_args=['-g', '-O0'],
    sources=[os.path.join("bson-numpy", "bsonnumpy.c")])

test_requires = []
test_suite = "test"
if sys.version_info[:2] == (2, 6):
    test_requires.append("unittest2")
    test_suite = "unittest2.collector"

setuptools.setup(
    name='BSON-NumPy',
    version='0.1',
    description='Module for converting directly from BSON to NumPy ndarrays'
                ' and vice versa',
    author='Anna Herlihy',
    author_email='anna@mongodb',
    url='',
    # long_description=readme_content,
    ext_modules=[bsonnumpymodule],
    test_suite=test_suite,
    tests_require=test_requires,
)
