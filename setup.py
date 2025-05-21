from setuptools import find_packages, setup

setup(
    name="shrunkiq",
    version="0.1.0",
    author="Chingis Owana",
    author_email="chingisoinar@gmail.com",
    description="ShrunkIQ Project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cowana-ai/ShrunkIQ",
    license="MIT",
    python_requires=">=3.9,<3.12",
    packages=find_packages(),
)
