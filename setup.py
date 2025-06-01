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
        ]
    },
    install_requires=[
        "numpy",
    ],
    python_requires=">=3.7",
)
