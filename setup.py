from setuptools import setup, find_packages

setup(
    name="pose6dof",
    version="0.1.0",
    description="A NumPy/SciPy-based SE(3) utility class",
    author="kshingent",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
