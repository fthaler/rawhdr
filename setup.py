# -*- coding: utf-8 -*-
"""Setup file for rawhdr."""

import os
import re
import setuptools


def read_file(*path):
    """Read file content."""
    package_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(package_path, *path), 'r') as text_file:
        return text_file.read()


def long_description():
    """Read long description from README.md."""
    return read_file('README.md')


def version():
    """Parse version info."""
    initfile_content = read_file('rawhdr', '__init__.py')
    match = re.search(r"__version__ = [']([^']*)[']", initfile_content, re.M)
    if match:
        return match.group(1)
    raise RuntimeError('Unable to find version string')


setuptools.setup(
    name='rawhdr',
    version=version(),
    author='Felix Thaler',
    author_email='felix.thaler@nummi.ch',
    description='A simple HDR image merger',
    long_description=long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/fthaler/rawhdr',
    packages=setuptools.find_packages(),
    scripts=['bin/rawhdrmerge'],
    install_requires=['click', 'imageio', 'numpy', 'rawpy'],
    classifiers=[
        'Programming Language :: Python', 'License :: OSI Approved :: '
        'GNU General Public License v2 or later (GPLv2+)',
        'Operating System :: OS Independent',
        'Topic :: Multimedia :: Graphics :: Graphics Conversion'
    ],
)
