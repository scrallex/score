from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "sep_quantum",
        sources=[
            "cpp/sep_quantum.cpp",
            "cpp/sep_core/core/qfh.cpp",
        ],
        include_dirs=[
            "cpp",
            "cpp/sep_core",
        ],
        libraries=["tbb"],
        cxx_std=20,
        language="c++",
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
