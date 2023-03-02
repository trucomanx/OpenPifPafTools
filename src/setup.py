#!/usr/bin/python

from setuptools import setup, find_packages

setup(
    name   ='OpenPifPafTools',
    version='0.1.0',
    author='Fernando Pujaico Rivera',
    author_email='fernando.pujaico.rivera@gmail.com',
    packages=['OpenPifPafTools'],
    #scripts=['bin/script1','bin/script2'],
    url='https://github.com/trucomanx/OpenPifPafTools',
    license='GPLv3',
    description='Tootl to work with OpenPifPaf',
    #long_description=open('README.txt').read(),
    install_requires=[
       "numpy", #"Django >= 1.1.1",
       "Pillow",
       "opencv-python",
       "openpifpaf"
    ],
)

#! python setup.py sdist bdist_wheel
# Upload to PyPi
# or 
#! pip3 install dist/OpenPifPafTools-0.1.tar.gz 
