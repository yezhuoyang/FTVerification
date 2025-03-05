from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="noise_sampler",  # Name of your module (can be anything)
    ext_modules=cythonize("noise_sampler.pyx"),  # Cythonize your .pyx file
    include_dirs=[np.get_include()],  # Include NumPy headers for compilation
    zip_safe=False,  # Prevent issues with C extensions
)