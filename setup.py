from setuptools import setup, find_packages

setup(
    name="motion_artifact_reduction",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)