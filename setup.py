from setuptools import setup, find_packages

setup(
    name="grid_tdhf",
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
        "HyQD-grid-methods @ git+https://github.com/HyQD/grid-methods.git@v1.0.0",
    ],
    python_requires=">=3.7",
)
