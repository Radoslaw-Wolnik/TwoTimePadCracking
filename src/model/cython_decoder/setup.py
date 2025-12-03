# setup.py for building the Cython extension
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


ext_modules = [
    Extension(
    "model.cython_decoder._twotimepad",
    ["src/model/cython_decoder/_twotimepad.pyx"],
    include_dirs=[np.get_include()],
    )
]

setup(
    name="model_cython_decoder",
    ext_modules=cythonize(ext_modules, compiler_directives={"language_level": "3"}),
)