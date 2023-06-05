"""Setup file for prototype template."""

import setuptools
import os

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

here = os.path.abspath(os.path.dirname(__file__))
version = {}
with open(os.path.join(here, "vqls_prototype", "__version__.py")) as f:
    exec(f.read(), version)

setuptools.setup(
    name="vqls_prototype",
    version=version["__version__"],
    description="Prototype for a Quantum Linear Solver",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=["Nicolas Renaud"],
    author_email="n.renaud@esciencecenter.nl",
    url="https://github.com/QuantumApplicationLab/vqls-prototype",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    python_requires=">=3.7",
    setup_requires=["setuptools_scm"],
    use_scm_version=False,
)
