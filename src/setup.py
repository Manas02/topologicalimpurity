from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy


ext_modules = [
    Extension(
        'tree',
        sources=['tree.pyx'],
        include_dirs=[numpy.get_include()],  # Add NumPy include path here
    ),
]

setup(
    name='tree',
    ext_modules=cythonize(ext_modules,
    compiler_directives={'language_level' : "3"}),
)
