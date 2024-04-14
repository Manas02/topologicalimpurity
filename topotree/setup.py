from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

# Define the extension modules
ext_modules = [
    Extension(
        'tree',
        sources=['tree.pyx'], 
        include_dirs=[numpy.get_include()], 
    ),
]

# Set up the package
setup(
    name='topotree',
    version='0.1.0',
    description='Topological Decision Tree and Random Forest Package',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/Manas02/topologicalimpurity',
    packages=find_packages(),  # Automatically finds all packages in the project
    ext_modules=cythonize(ext_modules,
                         compiler_directives={'language_level': "3"}),  # Compile Cython files
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Cython',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
    ],
)
