

from setuptools import setup
from Cython.Build import cythonize
import numpy as np


# Enable line profiling by adding profile=True
extensions = cythonize("noise_sampler.pyx", annotate=True, compiler_directives={'linetrace': True, 'profile': True})


setup(
    name="noise_sampler",  # Name of your module (can be anything)
    ext_modules=extensions,  # Cythonize your .pyx file
    include_dirs=[np.get_include()],  # Include NumPy headers for compilation
    zip_safe=False,  # Prevent issues with C extensions
)