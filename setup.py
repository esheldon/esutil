"""
Build the C/C++ extensions.  All other metadata is in pyproject.toml
"""
import sys
from glob import glob

import numpy
from setuptools import setup, Extension

include_dirs = [numpy.get_include(), "esutil/include"]

if sys.platform.startswith("win"):
    c_args = []
    cpp_args = []
else:
    c_args = ["-Wno-incompatible-pointer-types"]
    # The C++ code uses dynamic exception specifications, which were removed
    # in c++17, so we need to pin the standard
    cpp_args = c_args + ["-std=c++11"]

ext_modules = [
    Extension(
        "esutil.recfile._records",
        sources=[
            "esutil/recfile/records.cpp",
            "esutil/recfile/records_wrap.cpp",
        ],
        include_dirs=include_dirs + ["esutil/recfile"],
        extra_compile_args=cpp_args,
    ),
    Extension(
        "esutil.cosmology._cosmolib",
        sources=sorted(glob("esutil/cosmology/*.c")),
        include_dirs=include_dirs + ["esutil/cosmology"],
        extra_compile_args=c_args,
    ),
    Extension(
        "esutil.htm._htmc",
        sources=(
            sorted(glob("esutil/htm/htm_src/*.cpp"))
            + ["esutil/htm/htmc.cc", "esutil/htm/htmc_wrap.cc"]
        ),
        include_dirs=include_dirs + ["esutil/htm", "esutil/htm/htm_src"],
        extra_compile_args=cpp_args,
    ),
    Extension(
        "esutil.stat._chist",
        sources=["esutil/stat/chist_pywrap.c"],
        include_dirs=include_dirs,
        extra_compile_args=c_args,
    ),
    Extension(
        "esutil.integrate._cgauleg",
        sources=["esutil/integrate/cgauleg_pywrap.c"],
        include_dirs=include_dirs,
        extra_compile_args=c_args,
    ),
]

setup(ext_modules=ext_modules)
