from setuptools import setup, find_packages

setup(
    name="leap-policy",
    version="0.1.0",
    description="LeaP: Learnable source Prior for generative robot policies",
    packages=find_packages(include=["leap", "leap.*"]),
    python_requires=">=3.8",
    install_requires=["torch>=2.1", "torchvision>=0.16", "termcolor>=2.0", "numpy>=1.24,<2.0", "zarr>=2.12,<3"],
    extras_require={"robotwin": ["h5py>=3.8", "PyYAML>=6.0"]},
    license_files=["LICENSE", "licenses/*.txt", "THIRD_PARTY_NOTICES.md"],
)
