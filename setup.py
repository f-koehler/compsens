from setuptools import setup

setup(
    name="compsens",
    version="0.0.0",
    description="One-dimensional signal analysis using compressed sensing",
    url="https://github.com/f-koehler/compsens",
    author="Fabian Köhler",
    author_email="fkoehler1024@googlemail.com",
    license="MIT",
    install_requires=["cvxpy>=1.0.8", "numpy>=1.15.1", "scipy >= 1.1.0"],
    packages=["compsens"])
