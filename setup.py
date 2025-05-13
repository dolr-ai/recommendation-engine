from setuptools import setup, find_packages

setup(
    name="recommendation-engine",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
