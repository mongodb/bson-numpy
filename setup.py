import os
import sys
import numpy as np

import setuptools

bson_src = os.path.join("/usr", "local")
libraries = ["bson-1.0"]

bsonnumpymodule = setuptools.Extension('bsonnumpy',
                            define_macros=[('MAJOR_VERSION', '0'),
                                            ('MINOR_VERSION', '1')],
                            include_dirs=[os.path.join(bson_src, "include", "libbson-1.0"),
                                          np.get_include()],
                            library_dirs=[os.path.join(bson_src, "lib")],
                            libraries=libraries,
                            sources = [os.path.join("bson-numpy", "bsonnumpy.c")])

test_requires = []
test_suite = "test"
if sys.version_info[:2] == (2, 6):
    test_requires.append("unittest2")
    test_suite = "unittest2.collector"

setuptools.setup (name = 'BsonNumpy',
       version = '0.1',
       description = 'Module for converting directly from BSON to Numpy ndarrays and vice versa',
       author = 'Anna Herlihy',
       author_email = 'anna@mongodb',
       url = '',
       #long_description=readme_content,
       ext_modules = [bsonnumpymodule],
       test_suite=test_suite,
       tests_require=test_requires,
       )
