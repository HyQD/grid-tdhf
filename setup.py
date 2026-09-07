from setuptools import setup, find_packages

setup(
    name="HyQD-grid-tdhf",
    version="1.0.0",
    description="A grid-based time-dependent Hartree-Fock code",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "grid-tdhf = grid_tdhf.run:main",
            "grid-tdhf-mpi = grid_tdhf.parallel.run:main",
        ]
    },
    install_requires=[
        "numpy",
        "scipy",
        "sympy",
        "opt_einsum",
        "packaging",
        "tqdm",
        "numba",
        "HyQD-grid-lib @ git+https://github.com/HyQD/grid-lib.git@main",
    ],
    python_requires=">=3.7",
)
