from setuptools import setup, find_packages


setup(
    name='digitrecognition',
    version='0.1',
    description="Digit Recognition",
    author='Epameinondas Antonakos',
    author_email='antonakosn@gmail.com',
    packages=find_packages(),
    install_requires=['menpo>=0.7,<0.8']
)
