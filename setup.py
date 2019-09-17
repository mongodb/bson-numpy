import glob
import os
import subprocess
import sys

# Suppress warnings during shutdown, http://bugs.python.org/issue15881
try:
    import multiprocessing
except ImportError:
    pass

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext


# Single source the version.
version_file = os.path.realpath(os.path.join(
    os.path.dirname(__file__), 'bsonnumpy', 'version.py'))
version = {}
with open(version_file) as fp:
    exec(fp.read(), version)


try:
    from sphinx.setup_command import BuildDoc
    from sphinx.cmd import build as sphinxbuild
    HAVE_SPHINX = True
except Exception:
    HAVE_SPHINX = False


# Hack that ensures NumPy is installed prior to the build commencing.
# See http://stackoverflow.com/questions/19919905 for details.
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


CMDCLASS = {'build_ext': build_ext}


# Enables building docs and running doctests from setup.py
if HAVE_SPHINX:
    class build_sphinx(BuildDoc):

        description = "generate or test documentation"

        user_options = [("test", "t",
                         "run doctests instead of generating documentation")]

        boolean_options = ["test"]

        def initialize_options(self):
            self.test = False
            super().initialize_options()

        def run(self):
            # Run in-place build before Sphinx doc build.
            ret = subprocess.call(
                [sys.executable, sys.argv[0], 'build_ext', '-i'])
            if ret != 0:
                raise RuntimeError("Building BSON-Numpy failed!")

            if not HAVE_SPHINX:
                raise RuntimeError("You must install Sphinx to build or test "
                                   "the documentation.")

            if self.test:
                path = os.path.join(
                    os.path.abspath('.'), "doc", "_build", "doctest")
                mode = "doctest"
            else:
                path = os.path.join(
                    os.path.abspath('.'), "doc", "_build", version)
                mode = "html"

                try:
                    os.makedirs(path)
                except:
                    pass

            sphinx_args = ["-E", "-b", mode, "doc", path]
            status = sphinxbuild.main(sphinx_args)

            if status:
                raise RuntimeError("Documentation step '%s' failed" % (mode,))

            msg = "\nDocumentation step '{}' performed, results here:\n   {}\n"
            sys.stdout.write(msg.format(mode, path))

    CMDCLASS["doc"] = build_sphinx


def setup_package():
    with open('README.rst') as f:
        readme_content = f.read()

    build_requires = ["numpy>=1.17.0"]
    tests_require = build_requires + ["pymongo>=3.9.0,<4"]
    install_requires = build_requires + ["pymongo>=3.6.0,<4"]

    libraries = []
    if sys.platform == "win32":
        libraries.append("ws2_32")
    elif sys.platform != "darwin":
        # librt may be needed for clock_gettime()
        libraries.append("rt")

    setup(
        name='BSON-NumPy',
        version=version['__version__'],
        description='Module for converting directly from BSON to NumPy '
                    'ndarrays',
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
        setup_requires=build_requires,
        ext_modules=[
            Extension(
                'bsonnumpy._cbsonnumpy',
                sources=(glob.glob("bsonnumpy/*.c") +
                         glob.glob("bsonnumpy/*/*.c")),
                include_dirs=["bsonnumpy", "bsonnumpy/bson"],
                define_macros=[("BSON_COMPILATION", 1)],
                libraries=libraries)],
        install_requires=install_requires,
        test_suite="test",
        tests_require=tests_require,
        cmdclass=CMDCLASS,
        packages=["bsonnumpy"])


if __name__ == '__main__':
    setup_package()
