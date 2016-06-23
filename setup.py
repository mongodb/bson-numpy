import os
import numpy

from distutils.core import setup, Extension

bson_src = os.path.join("/usr", "local")
libraries = ["bson-1.0"]

bsonnumpymodule = Extension('bsonnumpy',
                            define_macros=[('MAJOR_VERSION', '0'),
                                            ('MINOR_VERSION', '1')],
                            include_dirs=[os.path.join(bson_src, "include", "libbson-1.0"),
                                          numpy.get_include()],
                            library_dirs=[os.path.join(bson_src, "lib")],
                            libraries=libraries,
                            sources = [os.path.join("bson-numpy", "bsonnumpy.c")])

setup (name = 'BsonNumpy',
       version = '0.1',
       description = 'Module for converting directly from BSON to Numpy ndarrays and vice versa',
       author = 'Anna Herlihy',
       author_email = 'anna@mongodb',
       url = '',
       #long_description=readme_content,
       ext_modules = [bsonnumpymodule])
